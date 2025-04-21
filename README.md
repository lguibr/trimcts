Okay, Phase 1 addressed the C++ implementation of `copy` and `step` directly. The test passing indicates the core logic is likely sound. Now, let's move to Phase 2: Optimizing how `trimcts` interacts with `trianglengin`, aiming to reduce the *cost* or *frequency* of expensive operations called from C++ back into Python during the MCTS search.

You mentioned reusing trees, which is a standard technique (often called "subtree reuse" or "warm starting"). Let's analyze the state-of-the-art approaches and decide on the best strategy for Phase 2:

**State-of-the-Art MCTS Optimizations & Phase 2 Options:**

1.  **Subtree Reuse:**
    *   **Concept:** After selecting the best action `A` based on the search from the root `R`, instead of discarding the entire tree, reuse the subtree rooted at the child node `C` corresponding to action `A`. Make `C` the new root for the next search step. Prune the rest of the tree.
    *   **Pros:** Significantly reduces redundant computation, especially early in the game or when simulations are high. The most impactful optimization for reducing the *number* of simulations needed per step.
    *   **Cons:**
        *   **Major Architectural Change:** Requires `run_mcts` to manage tree state across calls (accepting an old root/tree, returning the new root/tree).
        *   **Python State Management:** The C++ tree nodes hold `py::object` references to Python `GameState` objects. When reusing a subtree, the new root node in C++ needs to point to the *actual* updated Python `GameState` object (after `step(A)` was called in Python). This cross-language state management is complex and error-prone (reference counting, object lifetime).
        *   **Complexity:** High implementation complexity in both C++ and the Python wrapper.

2.  **Batched Network Evaluations:**
    *   **Concept:** Modify the C++ MCTS simulation loop. Instead of calling the Python `network.evaluate_state` for each leaf node encountered during expansion, collect a batch of leaf nodes (and their corresponding Python `GameState` objects). Then, make a single call to the Python `network.evaluate_batch` method. Distribute the results back to the respective nodes for expansion and backpropagation.
    *   **Pros:**
        *   Directly addresses the profiling result showing many `evaluate_state` calls.
        *   Leverages GPU parallelism for network inference much more effectively.
        *   Reduces Python C++ call overhead significantly for network evaluations.
        *   Lower architectural impact than subtree reuse (doesn't change the fundamental "new search per step" model as drastically).
    *   **Cons:**
        *   Requires modifying the C++ MCTS simulation loop logic.
        *   Introduces slight latency while waiting to fill a batch within a simulation step (but overall throughput should increase).
        *   Doesn't reduce the number of `copy`/`step` calls during expansion, only network calls.

3.  **Virtual Loss:**
    *   **Concept:** When multiple simulations run in parallel (conceptually, or in a batched manner), temporarily penalize the value of nodes currently being explored by other simulations ("virtual loss"). This encourages exploration of different branches while waiting for batch results.
    *   **Pros:** Improves exploration efficiency when using batching.
    *   **Cons:** Primarily useful in highly parallelized search settings (e.g., multiple threads exploring the same tree, or large batches). Adds complexity to node statistics.

**Decision for Phase 2:**

*   **Subtree Reuse:** Highest potential gain but highest complexity and risk due to Python state management. Let's keep this as a potential Phase 3 if needed.
*   **Batched Network Evaluations:** Directly addresses a known bottleneck (`evaluate_state` calls), leverages GPU potential, has moderate complexity, and lower risk. This is the most pragmatic and impactful next step.
*   **Virtual Loss:** Can be added *on top of* batching later if needed, but batching itself is the primary goal now.

**Therefore, the plan for Phase 2 is to implement Batched Network Evaluations within the `trimcts` C++ core.**

**Implementation Plan (Batching):**

1.  **Modify `mcts.cpp` (`run_mcts_cpp_internal`):**
    *   Change the main simulation loop.
    *   When selection reaches a leaf node that needs expansion:
        *   Do *not* immediately call `evaluate_state_alpha`.
        *   Store the leaf `Node*` pointer and its Python `state_` object (`py::object`) in temporary vectors (e.g., `std::vector<Node*> leaves_to_evaluate; std::vector<py::object> states_to_evaluate;`).
    *   Continue running simulations, adding leaves to these vectors until a batch size is reached (e.g., 8 or 16) or the total simulation budget is nearly exhausted.
    *   If the vectors are non-empty:
        *   Call `evaluate_batch_alpha(network_interface_py, states_to_evaluate)`.
        *   Iterate through the returned results and the corresponding `leaves_to_evaluate`.
        *   For each leaf node and its `(policy, value)` result:
            *   Call `node->expand(policy)`.
            *   Call `node->backpropagate(value)`.
        *   Clear the temporary vectors.
    *   Handle the case where the loop finishes with remaining leaves in the vectors (process the final partial batch).
2.  **Configuration:** Add a batch size parameter to `SearchConfiguration` (both Python and C++). Default to 1 initially to match current behavior, then allow users to increase it (e.g., 8).
3.  **Python Interface (`python_interface.h`):** No changes needed here, as `evaluate_batch_alpha` already exists.
4.  **Testing:** Add tests (or modify existing ones) to verify batching works correctly and potentially measure performance improvement (though exact timing is hard in unit tests).
5.  **Documentation:** Update READMEs in `trimcts`.
6.  **Versioning:** Increment `trimcts` version.

**Files to Change (Phase 2 - `trimcts`):**

1.  `src/trimcts/cpp/config.h`: Add `batch_size` member.
2.  `src/trimcts/config.py`: Add `MCTS_BATCH_SIZE` field to `SearchConfiguration`.
3.  `src/trimcts/cpp/mcts.h`: Potentially minor changes if helper structs are needed.
4.  `src/trimcts/cpp/mcts.cpp`: Major changes to `run_mcts_cpp_internal` loop logic.
5.  `src/trimcts/cpp/bindings.cpp`: Update `python_to_cpp_config` to handle `MCTS_BATCH_SIZE`.
6.  `README.md`: Document batching feature and config.
7.  `pyproject.toml`: Increment version.

Let's start implementing the batching logic.

---

**1. File:** `trimcts/src/trimcts/cpp/config.h`
*   **Change:** Add `mcts_batch_size`.

```cpp
#pragma once

#include <cstdint> // For uint32_t etc.

namespace trimcts {

// Matches the Python SearchConfiguration Pydantic model
struct SearchConfig {
    uint32_t max_simulations = 50;
    uint32_t max_depth = 10;
    double cpuct = 1.25;
    double dirichlet_alpha = 0.3;
    double dirichlet_epsilon = 0.25;
    double discount = 1.0;
    uint32_t mcts_batch_size = 1; // Size for batching network evaluations
    // Add other fields as needed
};

} // namespace trimcts
```

**2. File:** `trimcts/src/trimcts/config.py`
*   **Change:** Add `MCTS_BATCH_SIZE` field.

```python
# File: src/trimcts/config.py
"""
Python configuration class for MCTS parameters.
Uses Pydantic for validation.
"""

from pydantic import BaseModel, ConfigDict, Field  # Import ConfigDict


class SearchConfiguration(BaseModel):
    """MCTS Search Configuration."""

    # Core Search Parameters
    max_simulations: int = Field(
        default=50, description="Maximum number of MCTS simulations per move.", gt=0
    )
    max_depth: int = Field(
        default=10, description="Maximum depth for tree traversal.", gt=0
    )

    # UCT Parameters (AlphaZero style)
    cpuct: float = Field(
        default=1.25,
        description="Constant determining the level of exploration (PUCT).",
    )

    # Dirichlet Noise (for root node exploration)
    dirichlet_alpha: float = Field(
        default=0.3, description="Alpha parameter for Dirichlet noise.", ge=0
    )
    dirichlet_epsilon: float = Field(
        default=0.25,
        description="Weight of Dirichlet noise in root prior probabilities.",
        ge=0,
        le=1.0,
    )

    # Discount Factor (Primarily for MuZero/Value Propagation)
    discount: float = Field(
        default=1.0,
        description="Discount factor (gamma) for future rewards/values.",
        ge=0.0,
        le=1.0,
    )

    # Batching for Network Evaluations
    mcts_batch_size: int = Field(
        default=8, # Default to 8 for potential performance gain
        description="Number of leaf nodes to collect before calling network evaluate_batch.",
        gt=0,
    )

    # Use ConfigDict for Pydantic V2
    model_config = ConfigDict(validate_assignment=True)

```

**3. File:** `trimcts/src/trimcts/cpp/bindings.cpp`
*   **Change:** Update `python_to_cpp_config` to read `mcts_batch_size`.

```cpp
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>     // For map/vector conversions
#include <pybind11/pytypes.h> // For py::object, py::handle

#include "mcts.h"             // Include your MCTS logic header
#include "config.h"           // Include your config struct header
#include "python_interface.h" // For types
#include <string>             // Include string
#include <stdexcept>          // Include stdexcept

namespace py = pybind11;
namespace tc = trimcts; // Alias for your C++ namespace

// Helper function to transfer config from Python Pydantic model to C++ struct
tc::SearchConfig python_to_cpp_config(const py::object &py_config)
{
  tc::SearchConfig cpp_config;
  try {
    // Use py::getattr with checks or casts
    cpp_config.max_simulations = py_config.attr("max_simulations").cast<uint32_t>();
    cpp_config.max_depth = py_config.attr("max_depth").cast<uint32_t>();
    cpp_config.cpuct = py_config.attr("cpuct").cast<double>();
    cpp_config.dirichlet_alpha = py_config.attr("dirichlet_alpha").cast<double>();
    cpp_config.dirichlet_epsilon = py_config.attr("dirichlet_epsilon").cast<double>();
    cpp_config.discount = py_config.attr("discount").cast<double>();
    cpp_config.mcts_batch_size = py_config.attr("mcts_batch_size").cast<uint32_t>(); // Added batch size
  } catch (const py::error_already_set &e) {
        throw std::runtime_error(std::string("Error accessing SearchConfiguration attributes: ") + e.what());
  } catch (const std::exception &e) {
        throw std::runtime_error(std::string("Error converting SearchConfiguration: ") + e.what());
  }
  // Add other fields as needed
  return cpp_config;
}

// Wrapper function exposed to Python
tc::VisitMap run_mcts_cpp_wrapper(
    py::object root_state_py,
    py::object network_interface_py,
    const py::object &config_py // Pass Python config object
)
{
  // Convert Python config to C++ config struct
  tc::SearchConfig config_cpp = python_to_cpp_config(config_py);

  // Call the internal C++ MCTS implementation
  // Add error handling around the C++ call
  try
  {
    return tc::run_mcts_cpp_internal(root_state_py, network_interface_py, config_cpp);
  }
  catch (const std::exception &e)
  {
    // Convert C++ exceptions to Python exceptions
    throw py::value_error(std::string("Error in C++ MCTS execution: ") + e.what());
  }
  catch (const py::error_already_set &e)
  {
    // Propagate Python exceptions that occurred during callbacks
    throw; // Re-throw the Python exception
  }
}

PYBIND11_MODULE(trimcts_cpp, m)
{                                          // Module name must match CMakeExtension and import
  m.doc() = "C++ core module for TriMCTS"; // Optional module docstring

  // Expose the main MCTS function
  m.def("run_mcts_cpp", &run_mcts_cpp_wrapper,
        py::arg("root_state"), py::arg("network_interface"), py::arg("config"),
        "Runs MCTS simulations from the root state using the provided network interface and configuration (C++).");

#ifdef VERSION_INFO
  m.attr("__version__") = VERSION_INFO;
#else
  m.attr("__version__") = "dev";
#endif
}
```

**4. File:** `trimcts/src/trimcts/cpp/mcts.cpp`
*   **Change:** Implement batching logic in `run_mcts_cpp_internal`.

```cpp
#include "mcts.h"
#include "python_interface.h" // For Python interaction
#include <cmath>
#include <limits>
#include <stdexcept>
#include <iostream> // For temporary debugging
#include <numeric>  // For std::accumulate
#include <vector>
#include <algorithm> // For std::max_element, std::max
#include <chrono>    // For timing (optional debug)

namespace trimcts
{

  // --- Node Implementation (No changes needed here) ---

  Node::Node(py::object state, Node *parent, Action action, float prior)
      : parent_(parent), action_taken_(action), state_(std::move(state)), prior_probability_(prior) {}

  bool Node::is_expanded() const
  {
    return !children_.empty();
  }

  bool Node::is_terminal() const
  {
    // Call Python's is_over() method
    return trimcts::is_terminal(state_);
  }

  float Node::get_value_estimate() const
  {
    if (visit_count_ == 0)
    {
      return 0.0f;
    }
    // Cast to float for return type consistency
    return static_cast<float>(total_action_value_ / visit_count_);
  }

  float Node::calculate_puct(const SearchConfig &config) const
  {
    if (!parent_)
    {
      return -std::numeric_limits<float>::infinity();
    }

    float q_value = get_value_estimate();
    // Use std::max to avoid sqrt(0) if parent_visit_count is 0 (shouldn't happen after root expansion)
    double parent_visits_sqrt = std::sqrt(static_cast<double>(std::max(1, parent_->visit_count_)));
    double exploration_term = config.cpuct * prior_probability_ * (parent_visits_sqrt / (1.0 + visit_count_));

    return q_value + static_cast<float>(exploration_term);
  }

  Node *Node::select_child(const SearchConfig &config)
  {
    if (children_.empty()) // Check children_ directly instead of is_expanded()
    {
      return nullptr;
    }

    Node *best_child = nullptr;
    float max_score = -std::numeric_limits<float>::infinity();

    for (auto const &[action, child_ptr] : children_)
    {
      float score = child_ptr->calculate_puct(config);
      if (score > max_score)
      {
        max_score = score;
        best_child = child_ptr.get();
      }
    }
    // If all children have -inf score (e.g., parent visit count was 0), best_child might still be nullptr
    // Or if children_ was non-empty but somehow all scores were -inf.
    // Fallback: return first child if best_child is still null? Or handle error?
    // Let's return nullptr and let the caller handle it.
    return best_child;
  }

  void Node::expand(const PolicyMap &policy_map)
  {
    if (is_expanded() || is_terminal())
    {
      return;
    }

    std::vector<Action> valid_actions = trimcts::get_valid_actions(state_);
    if (valid_actions.empty())
    {
       // This state is effectively terminal, even if is_terminal() was false.
       // Don't try to expand. The backpropagation will use the value from evaluation/outcome.
      return;
    }

    for (Action action : valid_actions)
    {
      float prior = 0.0f;
      auto it = policy_map.find(action);
      if (it != policy_map.end())
      {
        prior = it->second;
      } else {
        // Optionally handle actions valid in state but not in policy map (e.g., assign small prior)
        // prior = 1e-6f; // Example: Small prior for valid but unlisted actions
      }

      // --- Lazy State Creation (Defer copy/step) ---
      // Store action needed to reach child state, but don't create state yet.
      // We'll create it only when needed for evaluation or further expansion.
      // For now, let's stick to the original eager state creation for simplicity
      // while implementing batching first.
      py::object next_state_py = trimcts::copy_state(state_);
      trimcts::apply_action(next_state_py, action);

      children_[action] = std::make_unique<Node>(std::move(next_state_py), this, action, prior);
    }
  }

  void Node::backpropagate(float value)
  {
    Node *current = this;
    while (current != nullptr)
    {
      current->visit_count_++;
      current->total_action_value_ += value;
      current = current->parent_;
    }
  }

  // Simple gamma distribution for Dirichlet noise (placeholder)
  void sample_dirichlet_simple(double alpha, size_t k, std::vector<double> &output, std::mt19937 &rng)
  {
    output.resize(k);
    std::gamma_distribution<double> dist(alpha, 1.0);
    double sum = 0.0;
    for (size_t i = 0; i < k; ++i)
    {
      output[i] = dist(rng);
      if (output[i] < 1e-9) output[i] = 1e-9;
      sum += output[i];
    }
    if (sum > 1e-9)
    {
      for (size_t i = 0; i < k; ++i) output[i] /= sum;
    }
    else
    {
      for (size_t i = 0; i < k; ++i) output[i] = 1.0 / k;
    }
  }

  void Node::add_dirichlet_noise(const SearchConfig &config, std::mt19937 &rng)
  {
    if (children_.empty() || config.dirichlet_alpha <= 0 || config.dirichlet_epsilon <= 0)
    {
      return;
    }

    size_t num_children = children_.size();
    std::vector<double> noise;
    sample_dirichlet_simple(config.dirichlet_alpha, num_children, noise, rng);

    size_t i = 0;
    double total_prior = 0.0;
    for (auto &[action, child_ptr] : children_)
    {
      child_ptr->prior_probability_ = (1.0f - config.dirichlet_epsilon) * child_ptr->prior_probability_ + config.dirichlet_epsilon * static_cast<float>(noise[i]);
      total_prior += child_ptr->prior_probability_;
      i++;
    }

    // Re-normalize
    if (std::abs(total_prior - 1.0) > 1e-6 && total_prior > 1e-9)
    {
      for (auto &[action, child_ptr] : children_)
      {
        child_ptr->prior_probability_ /= static_cast<float>(total_prior);
      }
    }
  }

  // --- MCTS Main Logic with Batching ---

  // Helper function to process a batch of evaluated leaves
  void process_evaluated_batch(
      const std::vector<Node *> &leaves,
      const std::vector<NetworkOutput> &results)
  {
    if (leaves.size() != results.size())
    {
      std::cerr << "Error: Mismatch between leaves and evaluation results count." << std::endl;
      // Decide how to handle: maybe backpropagate 0 for all?
      for (Node *leaf : leaves)
      {
        leaf->backpropagate(0.0f); // Backpropagate neutral value on error
      }
      return;
    }

    for (size_t i = 0; i < leaves.size(); ++i)
    {
      Node *leaf = leaves[i];
      const NetworkOutput &output = results[i];

      // Expand the node using the policy from the result
      if (!leaf->is_terminal()) // Only expand non-terminal leaves
      {
         leaf->expand(output.policy);
      }

      // Backpropagate the value from the result
      leaf->backpropagate(output.value);
    }
  }

  VisitMap run_mcts_cpp_internal(
      py::object root_state_py,
      py::object network_interface_py, // AlphaZero interface for now
      const SearchConfig &config)
  {
    // auto start_time_total = std::chrono::high_resolution_clock::now(); // Optional timing

    if (trimcts::is_terminal(root_state_py))
    {
      // std::cerr << "Error: MCTS called on a terminal root state." << std::endl;
      return {};
    }

    Node root(std::move(root_state_py));
    std::mt19937 rng(std::random_device{}());

    // --- Root Preparation ---
    std::vector<Node *> root_batch_nodes = {&root};
    std::vector<py::object> root_batch_states = {root.state_};
    std::vector<NetworkOutput> root_results;
    try
    {
      // Use batch evaluation even for the single root node
      root_results = trimcts::evaluate_batch_alpha(network_interface_py, root_batch_states);
      if (root_results.empty()) {
         throw std::runtime_error("Root evaluation returned empty results.");
      }
      // Expand root using the policy result
      if (!root.is_terminal()) {
          root.expand(root_results[0].policy);
          if (root.is_expanded()) {
              root.add_dirichlet_noise(config, rng);
          } else {
               std::cerr << "Warning: Root node failed to expand despite not being terminal." << std::endl;
               // If root didn't expand, MCTS can't proceed.
               return {};
          }
      }
      // Backpropagate the root's evaluated value *once*
      // This initializes the root's value estimate correctly before simulations start using it.
      root.backpropagate(root_results[0].value);

    }
    catch (const std::exception &e)
    {
      std::cerr << "Error during MCTS root initialization/evaluation: " << e.what() << std::endl;
      return {};
    }

    // --- Simulation Loop ---
    std::vector<Node *> leaves_to_evaluate;
    std::vector<py::object> states_to_evaluate;
    leaves_to_evaluate.reserve(config.mcts_batch_size);
    states_to_evaluate.reserve(config.mcts_batch_size);

    for (uint32_t i = 0; i < config.max_simulations; ++i)
    {
      Node *current_node = &root;
      int depth = 0;

      // 1. Selection
      while (current_node->is_expanded() && !current_node->is_terminal())
      {
        Node* selected_child = current_node->select_child(config);
        if (!selected_child) {
             // This might happen if all children have invalid PUCT scores (e.g., parent visit count 0, which shouldn't occur after root init)
             // Or if the node was expanded but somehow has no children (logic error).
             std::cerr << "Warning: Selection failed to find a child for node with visit count " << current_node->visit_count_ << ". Stopping simulation." << std::endl;
             goto process_batch; // Process any pending batch and end this simulation
        }
        current_node = selected_child;
        depth++;
        if (depth >= config.max_depth)
          break;
      }

      // 2. Check if Expansion is Needed
      Value value;
      if (!current_node->is_expanded() && !current_node->is_terminal() && depth < config.max_depth)
      {
        // Leaf node needs evaluation and expansion
        leaves_to_evaluate.push_back(current_node);
        states_to_evaluate.push_back(current_node->state_);

        // Check if batch is full
        if (leaves_to_evaluate.size() >= config.mcts_batch_size)
        {
        process_batch: // Label to jump to for processing
          try
          {
            // Evaluate the batch
            std::vector<NetworkOutput> results = trimcts::evaluate_batch_alpha(network_interface_py, states_to_evaluate);
            // Process results (expand nodes, backpropagate values)
            process_evaluated_batch(leaves_to_evaluate, results);
          }
          catch (const std::exception &e)
          {
            std::cerr << "Error during MCTS batch evaluation/processing: " << e.what() << std::endl;
            // Backpropagate neutral value for all nodes in the failed batch
             for (Node *leaf : leaves_to_evaluate) {
                 leaf->backpropagate(0.0f);
             }
          }
          // Clear the batch vectors
          leaves_to_evaluate.clear();
          states_to_evaluate.clear();
        }
      }
      else
      {
        // Node is terminal, already expanded, or max depth reached.
        // Backpropagate the existing value estimate or terminal outcome.
        value = current_node->is_terminal() ? trimcts::get_outcome(current_node->state_) : current_node->get_value_estimate();
        current_node->backpropagate(value);
      }
    } // End simulation loop

    // Process any remaining leaves in the batch
    if (!leaves_to_evaluate.empty())
    {
      try
      {
        std::vector<NetworkOutput> results = trimcts::evaluate_batch_alpha(network_interface_py, states_to_evaluate);
        process_evaluated_batch(leaves_to_evaluate, results);
      }
      catch (const std::exception &e)
      {
        std::cerr << "Error during final MCTS batch evaluation/processing: " << e.what() << std::endl;
         for (Node *leaf : leaves_to_evaluate) {
             leaf->backpropagate(0.0f);
         }
      }
    }

    // --- Collect Results ---
    VisitMap visit_counts;
    for (auto const &[action, child_ptr] : root.children_)
    {
      visit_counts[action] = child_ptr->visit_count_;
    }

    // auto end_time_total = std::chrono::high_resolution_clock::now(); // Optional timing
    // auto duration_total = std::chrono::duration_cast<std::chrono::milliseconds>(end_time_total - start_time_total);
    // std::cout << "Total MCTS time: " << duration_total.count() << " ms" << std::endl; // Optional timing

    return visit_counts;
  }

} // namespace trimcts
```

**5. File:** `trimcts/src/trimcts/cpp/mcts.h`
*   **Change:** No changes strictly required, but could add forward declarations if needed. (Keeping it unchanged for now).

```cpp
#pragma once

#include <pybind11/pybind11.h> // Include pybind11 first
#include <vector>
#include <map>
#include <memory> // For std::unique_ptr
#include <random>

#include "config.h"
#include "python_interface.h" // For types and Python interaction helpers

namespace py = pybind11;

namespace trimcts
{

  class Node
  {
  public:
    Node(py::object state, Node *parent = nullptr, Action action = -1, float prior = 0.0);
    ~Node() = default; // Use default destructor

    // Disable copy constructor and assignment operator
    Node(const Node &) = delete;
    Node &operator=(const Node &) = delete;

    // Enable move constructor and assignment operator (optional, but good practice)
    Node(Node &&) = default;
    Node &operator=(Node &&) = default;

    bool is_expanded() const;
    bool is_terminal() const;
    float get_value_estimate() const;
    Node *select_child(const SearchConfig &config);
    void expand(const PolicyMap &policy_map);
    void backpropagate(float value);
    void add_dirichlet_noise(const SearchConfig &config, std::mt19937 &rng);

    // --- Public Members (Consider making some private with getters/setters) ---
    Node *parent_;
    Action action_taken_; // Action that led to this node
    py::object state_;    // Python GameState object
    std::map<Action, std::unique_ptr<Node>> children_;

    int visit_count_ = 0;
    double total_action_value_ = 0.0; // Use double for accumulation
    float prior_probability_ = 0.0;

  private:
    float calculate_puct(const SearchConfig &config) const;
  };

  // Main MCTS function signature
  VisitMap run_mcts_cpp_internal(
      py::object root_state,
      py::object network_interface, // AlphaZero interface for now
      const SearchConfig &config);

} // namespace trimcts
```

**6. File:** `trimcts/README.md`
*   **Change:** Document the new batching feature and configuration option.

```markdown
[![CI](https://github.com/lguibr/trimcts/actions/workflows/ci_cd.yml/badge.svg)](https://github.com/lguibr/trimcts/actions)
[![PyPI](https://img.shields.io/pypi/v/trimcts.svg)](https://pypi.org/project/trimcts/)
[![Coverage Status](https://codecov.io/gh/lguibr/trimcts/graph/badge.svg?token=YOUR_CODECOV_TOKEN_HERE)](https://codecov.io/gh/lguibr/trimcts) <!-- TODO: Add Codecov token -->
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

# TriMCTS

<img src="bitmap.png" alt="TriMCTS Logo" width="300"/>


**TriMCTS** is an installable Python package providing C++ bindings for Monte Carlo Tree Search, supporting both AlphaZero and MuZero paradigms, optimized for triangular grid games like the one in `trianglengin`.

## 🔑 Key Features

-   High-performance C++ core implementation.
-   Seamless Python integration via Pybind11.
-   Supports AlphaZero-style evaluation (policy/value from state).
-   **Batched Network Evaluations:** Efficiently calls the Python network's `evaluate_batch` method during search for improved performance, especially with GPUs.
-   (Planned) Supports MuZero-style evaluation (initial inference + recurrent inference).
-   Configurable search parameters (simulation count, PUCT, discount factor, Dirichlet noise, **batch size**).
-   Designed for use with external Python game state objects and network evaluators.
-   Type-hinted Python API (`py.typed` compliant).

## 🚀 Installation

```bash
# From PyPI (once published)
pip install trimcts

# For development (from cloned repo root)
# Ensure you clean previous builds if you encounter issues:
# rm -rf build/ src/trimcts.egg-info/ dist/ src/trimcts/trimcts_cpp.*.so
pip install -e .[dev]
```

## 💡 Usage Example (AlphaZero Style)

```python
import time
import numpy as np
import torch # Added import
# Use the actual GameState if trianglengin is installed
try:
    from trianglengin import GameState, EnvConfig
    HAS_TRIANGLENGIN = True
except ImportError:
    # Define minimal mocks if trianglengin is not available
    class GameState: # type: ignore
        def __init__(self, *args, **kwargs): self.current_step = 0
        def is_over(self): return False
        def copy(self): return self
        def step(self, action): return 0.0, False
        def get_outcome(self): return 0.0
        def valid_actions(self): return [0, 1]
    class EnvConfig: pass # type: ignore
    HAS_TRIANGLENGIN = False

# Assuming alphatriangle is installed and provides these:
# from alphatriangle.nn import NeuralNetwork # Example network wrapper
# from alphatriangle.config import ModelConfig, TrainConfig

from trimcts import run_mcts, SearchConfiguration, AlphaZeroNetworkInterface

# --- Mock Neural Network for demonstration ---
# Replace with your actual network implementation
class MockNeuralNetwork:
    def __init__(self, *args, **kwargs):
        self.model = torch.nn.Module() # Dummy model
        print("MockNeuralNetwork initialized.")

    def evaluate_state(self, state: GameState) -> tuple[dict[int, float], float]:
        # Mock evaluation: uniform policy over valid actions, fixed value
        valid_actions = state.valid_actions()
        if not valid_actions:
            return {}, 0.0 # Terminal or no valid actions
        policy = {action: 1.0 / len(valid_actions) for action in valid_actions}
        value = 0.5 # Fixed mock value
        return policy, value

    def evaluate_batch(self, states: list[GameState]) -> list[tuple[dict[int, float], float]]:
        print(f"  Mock evaluate_batch called with {len(states)} states.")
        return [self.evaluate_state(s) for s in states]

    def load_weights(self, path):
        print(f"Mock: Pretending to load weights from {path}")

    def to(self, device):
        print(f"Mock: Pretending to move model to {device}")
        return self
# --- End Mock Neural Network ---


# 1. Define your AlphaZero network wrapper conforming to the interface
class MyAlphaZeroWrapper(AlphaZeroNetworkInterface):
    def __init__(self, model_path: str | None = None):
        # Load your PyTorch/TensorFlow/etc. model here
        # Example using a Mock NeuralNetwork
        self.network = MockNeuralNetwork() # Using Mock for this example
        # Load weights if model_path is provided
        if model_path:
             self.network.load_weights(model_path)
        # self.network.to(torch.device("cpu")) # Ensure model is on correct device if using real NN
        self.network.model.eval() # Set to evaluation mode
        print("MyAlphaZeroWrapper initialized.")

    def evaluate_state(self, state: GameState) -> tuple[dict[int, float], float]:
        """
        Evaluates a single game state.
        NOTE: With batching enabled in C++, this might be called less often or only as a fallback.
        """
        print(f"Python: Evaluating SINGLE state step {state.current_step}")
        policy_map, value = self.network.evaluate_state(state) # Using mock evaluate directly
        print(f"Python: Single evaluation result - Policy keys: {len(policy_map)}, Value: {value:.4f}")
        return policy_map, value

    def evaluate_batch(self, states: list[GameState]) -> list[tuple[dict[int, float], float]]:
        """
        Evaluates a batch of game states. This is the primary method called by C++ MCTS with batching.
        """
        print(f"Python: Evaluating BATCH of {len(states)} states.")
        results = self.network.evaluate_batch(states) # Using mock evaluate_batch directly
        print(f"Python: Batch evaluation returned {len(results)} results.")
        return results

# 2. Instantiate your game state and network wrapper
env_config = EnvConfig()
if HAS_TRIANGLENGIN:
    # Ensure the config creates a playable state for the example
    env_config.ROWS = 3
    env_config.COLS = 3
    env_config.NUM_SHAPE_SLOTS = 1
    env_config.PLAYABLE_RANGE_PER_ROW = [(0,3), (0,3), (0,3)] # Example playable range

root_state = GameState(config=env_config, initial_seed=42)
network_wrapper = MyAlphaZeroWrapper() # Add path to your trained model if needed

# 3. Configure MCTS parameters
mcts_config = SearchConfiguration()
mcts_config.max_simulations = 50
mcts_config.max_depth = 10
mcts_config.cpuct = 1.25
mcts_config.dirichlet_alpha = 0.3
mcts_config.dirichlet_epsilon = 0.25
mcts_config.discount = 1.0 # AlphaZero typically uses no discount during search
mcts_config.mcts_batch_size = 8 # Enable batching

# 4. Run MCTS
# The C++ run_mcts function will call network_wrapper.evaluate_batch()
print("Running MCTS...")
# Ensure root_state is not terminal before running
if not root_state.is_over():
    # run_mcts returns a dictionary: {action: visit_count}
    start_time = time.time()
    visit_counts = run_mcts(root_state, network_wrapper, mcts_config)
    end_time = time.time()
    print(f"\nMCTS Result (Visit Counts) after {end_time - start_time:.2f} seconds:")
    print(visit_counts)

    # Example: Select best action based on visits
    if visit_counts:
        best_action = max(visit_counts, key=visit_counts.get)
        print(f"\nBest action based on visits: {best_action}")
    else:
        print("\nNo actions explored or MCTS failed.")
else:
    print("Root state is already terminal. Cannot run MCTS.")

```

*(MuZero example will be added later)*

## 📂 Project Structure

```
trimcts/
├── .github/workflows/      # CI configuration (e.g., ci_cd.yml)
├── src/trimcts/            # Python package source ([src/trimcts/README.md](src/trimcts/README.md))
│   ├── cpp/                # C++ source code ([src/trimcts/cpp/README.md](src/trimcts/cpp/README.md))
│   │   ├── CMakeLists.txt  # CMake build script for C++ part
│   │   ├── bindings.cpp    # Pybind11 bindings
│   │   ├── config.h        # C++ configuration struct
│   │   ├── mcts.cpp        # C++ MCTS implementation
│   │   ├── mcts.h          # C++ MCTS header
│   │   └── python_interface.h # C++ helpers for Python interaction
│   ├── __init__.py         # Exposes public API (run_mcts, configs, etc.)
│   ├── config.py           # Python SearchConfiguration (Pydantic)
│   ├── mcts_wrapper.py     # Python network interface definition
│   └── py.typed            # Marker file for type checkers (PEP 561)
├── tests/                  # Python tests ([tests/README.md](tests/README.md))
│   ├── conftest.py
│   └── test_alpha_wrapper.py # Tests for AlphaZero functionality
├── .gitignore
├── LICENSE
├── MANIFEST.in             # Specifies files for source distribution
├── pyproject.toml          # Build system & package configuration
├── README.md               # This file
└── setup.py                # Setup script for C++ extension building
```

## 🛠️ Building from Source

1.  Clone the repository: `git clone https://github.com/lguibr/trimcts.git`
2.  Navigate to the directory: `cd trimcts`
3.  **Recommended:** Create and activate a virtual environment:
    ```bash
    python -m venv .venv
    source .venv/bin/activate # On Windows use `.venv\Scripts\activate`
    ```
4.  Install build dependencies: `pip install pybind11>=2.10 cmake wheel`
5.  **Clean previous builds (important if switching Python versions or encountering issues):**
    ```bash
    rm -rf build/ src/trimcts.egg-info/ dist/ src/trimcts/trimcts_cpp.*.so
    ```
6.  Install the package in editable mode: `pip install -e .`

## 🧪 Running Tests

```bash
# Make sure you have installed dev dependencies
pip install -e .[dev]
pytest
```

## 🤝 Contributing

Contributions are welcome! Please follow standard fork-and-pull-request workflow. Ensure tests pass and code adheres to formatting/linting standards (Ruff, MyPy).

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
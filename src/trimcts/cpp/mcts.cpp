// File: trimcts/src/trimcts/cpp/mcts.cpp
#include "mcts.h"
#include "mcts_manager.h"     // Include the manager
#include "python_interface.h" // For Python interaction
#include <cmath>
#include <limits>
#include <stdexcept>
#include <iostream>
#include <numeric>
#include <vector>
#include <algorithm>
#include <chrono>
#include <utility> // For std::pair, std::move
#include <tuple>   // For std::tuple return type

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace trimcts
{

  // --- Node Implementation ---

  // Constructor for non-root nodes
  Node::Node(Node *parent, Action action, float prior)
      : parent_(parent), action_taken_(action), prior_probability_(prior),
        materialized_state_(std::nullopt), state_materialized_(false) {}

  // Special method to set the state for the root node
  void Node::set_root_state(py::object state)
  {
    materialized_state_ = std::move(state);
    state_materialized_ = true;
    parent_ = nullptr;  // Root has no parent
    action_taken_ = -1; // No action led to root
  }

  // Get or materialize the state
  py::object Node::get_state()
  {
    // If state is already materialized, return it
    if (state_materialized_ && materialized_state_.has_value())
    {
      return *materialized_state_;
    }

    // If this is the root node but state wasn't set (error condition)
    if (!parent_)
    {
      throw std::logic_error("Attempted to get_state for root node before state was set.");
    }

    // Materialize the state: Get parent's state, copy, apply action
    // This is the potentially expensive part
    // std::cout << "[DEBUG] Materializing state for node reached by action " << action_taken_ << std::endl; // Debug log
    try
    {
      py::object parent_state = parent_->get_state(); // Recursive call
      py::object my_state = trimcts::copy_state(parent_state);
      trimcts::apply_action(my_state, action_taken_);

      materialized_state_ = std::move(my_state);
      state_materialized_ = true;
      return *materialized_state_;
    }
    catch (const std::exception &e)
    {
      std::cerr << "Error materializing state for node reached by action " << action_taken_ << ": " << e.what() << std::endl;
      throw; // Re-throw exception
    }
    catch (const py::error_already_set &e)
    {
      std::cerr << "Python error materializing state for node reached by action " << action_taken_ << ": " << e.what() << std::endl;
      throw; // Re-throw Python exception
    }
  }

  bool Node::has_state_materialized() const
  {
    return state_materialized_;
  }

  Node *Node::get_parent() const
  {
    return parent_;
  }
  Action Node::get_action_taken() const
  {
    return action_taken_;
  }

  bool Node::is_expanded() const { return !children_.empty(); }

  // is_terminal now needs to potentially materialize the state
  bool Node::is_terminal()
  {
    py::object state = get_state(); // Ensure state is available
    return trimcts::is_terminal(state);
  }

  float Node::get_value_estimate() const
  {
    if (visit_count_ == 0)
      return 0.0f;
    return visit_count_ > 0 ? static_cast<float>(total_action_value_ / visit_count_) : 0.0f;
  }

  float Node::calculate_puct(const SearchConfig &config) const
  {
    if (!parent_)
      return std::numeric_limits<float>::infinity();

    float q_value = get_value_estimate();
    double parent_visits_sqrt = std::sqrt(static_cast<double>(std::max(1, parent_->visit_count_)));
    double exploration_term = config.cpuct * prior_probability_ * (parent_visits_sqrt / (1.0 + visit_count_));

    return q_value + static_cast<float>(exploration_term);
  }

  Node *Node::select_child(const SearchConfig &config)
  {
    if (children_.empty())
      return nullptr;
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
    if (best_child == nullptr && !children_.empty())
    {
      std::cerr << "Warning: select_child failed to find a best child despite having children." << std::endl;
    }
    return best_child;
  }

  // Expansion no longer creates state, just child nodes
  void Node::expand(const PolicyMap &policy_map)
  {
    if (is_expanded() || is_terminal()) // is_terminal() might materialize state here
      return;

    py::object current_state = get_state(); // Ensure state is available for get_valid_actions
    std::vector<Action> valid_actions = trimcts::get_valid_actions(current_state);
    if (valid_actions.empty())
      return; // Cannot expand if no valid actions

    for (Action action : valid_actions)
    {
      float prior = 0.0f;
      auto it = policy_map.find(action);
      if (it != policy_map.end())
        prior = it->second;

      // Create child node without materializing its state yet
      children_[action] = std::make_unique<Node>(this, action, prior);
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

  std::unique_ptr<Node> Node::detach_child(Action action)
  {
    auto it = children_.find(action);
    if (it == children_.end())
    {
      return nullptr;
    }
    std::unique_ptr<Node> child_ptr = std::move(it->second);
    children_.erase(it);
    return child_ptr;
  }

  // Updates the Python state object held by this node.
  // This is primarily used when reusing a node as the new root.
  void Node::update_state(py::object new_state)
  {
    materialized_state_ = std::move(new_state);
    state_materialized_ = true; // Mark as materialized
  }

  void Node::set_parent(Node *new_parent)
  {
    parent_ = new_parent;
  }

  // sample_dirichlet_simple remains the same
  void sample_dirichlet_simple(double alpha, size_t k, std::vector<double> &output, std::mt19937 &rng)
  {
    output.resize(k);
    std::gamma_distribution<double> dist(alpha, 1.0);
    double sum = 0.0;
    for (size_t i = 0; i < k; ++i)
    {
      output[i] = dist(rng);
      if (output[i] < 1e-9)
        output[i] = 1e-9;
      sum += output[i];
    }
    if (sum > 1e-9)
    {
      for (size_t i = 0; i < k; ++i)
        output[i] /= sum;
    }
    else
    {
      if (k > 0)
      {
        for (size_t i = 0; i < k; ++i)
          output[i] = 1.0 / k;
      }
    }
  }

  void Node::add_dirichlet_noise(const SearchConfig &config, std::mt19937 &rng)
  {
    if (children_.empty() || config.dirichlet_alpha <= 0 || config.dirichlet_epsilon <= 0)
      return;
    size_t num_children = children_.size();
    std::vector<double> noise;
    sample_dirichlet_simple(config.dirichlet_alpha, num_children, noise, rng);
    size_t i = 0;
    double total_prior = 0.0;
    for (auto &[action, child_ptr] : children_)
    {
      float current_prior = child_ptr->prior_probability_;
      if (!std::isfinite(current_prior))
      {
        std::cerr << "Warning: Non-finite prior probability encountered before adding noise for action " << action << ". Resetting to 0." << std::endl;
        current_prior = 0.0f;
      }
      child_ptr->prior_probability_ = (1.0f - static_cast<float>(config.dirichlet_epsilon)) * current_prior + static_cast<float>(config.dirichlet_epsilon * noise[i]);
      total_prior += child_ptr->prior_probability_;
      i++;
    }
    if (std::abs(total_prior - 1.0) > 1e-6 && total_prior > 1e-9)
    {
      for (auto &[action, child_ptr] : children_)
      {
        child_ptr->prior_probability_ /= static_cast<float>(total_prior);
      }
    }
    else if (total_prior <= 1e-9 && num_children > 0)
    {
      float uniform_prior = 1.0f / static_cast<float>(num_children);
      for (auto &[action, child_ptr] : children_)
      {
        child_ptr->prior_probability_ = uniform_prior;
      }
      std::cerr << "Warning: Total prior probability near zero after adding noise. Resetting to uniform." << std::endl;
    }
  }

  // --- MCTS Main Logic with Lazy State Creation ---

  // process_evaluated_batch needs to get state before expanding
  void process_evaluated_batch(
      const std::vector<Node *> &leaves,
      const std::vector<NetworkOutput> &results)
  {
    if (leaves.size() != results.size())
    {
      std::cerr << "Error: Mismatch between leaves (" << leaves.size()
                << ") and evaluation results (" << results.size()
                << ") count." << std::endl;
      for (Node *leaf : leaves)
        leaf->backpropagate(0.0f);
      return;
    }
    for (size_t i = 0; i < leaves.size(); ++i)
    {
      Node *leaf = leaves[i];
      const NetworkOutput &output = results[i];
      // Expand *before* backpropagating value from evaluation
      if (!leaf->is_terminal())
      { // is_terminal ensures state is materialized
        try
        {
          leaf->expand(output.policy);
        }
        catch (const std::exception &e)
        {
          std::cerr << "Error during leaf expansion: " << e.what() << std::endl;
        }
        catch (const py::error_already_set &e)
        {
          std::cerr << "Python error during leaf expansion: " << e.what() << std::endl;
        }
      }
      // Backpropagate the value from the network evaluation
      leaf->backpropagate(output.value);
    }
  }

  // Main MCTS function updated for tree reuse and lazy state
  // Returns tuple: (VisitMap, capsule_handle, avg_depth)
  PYBIND11_EXPORT std::tuple<VisitMap, py::capsule, double> run_mcts_cpp_internal(
      py::object current_root_state_py,
      py::object network_interface_py,
      const SearchConfig &config,
      const py::capsule &previous_tree_capsule,
      Action last_action)
  {
    std::unique_ptr<MctsTreeManager> tree_manager_ptr;
    Node *root_node = nullptr;
    bool reused_tree = false;

    // --- Tree Reuse Logic ---
    if (previous_tree_capsule && PyCapsule_IsValid(previous_tree_capsule.ptr(), "MctsTreeManager"))
    {
      MctsTreeManager *previous_manager = static_cast<MctsTreeManager *>(
          PyCapsule_GetPointer(previous_tree_capsule.ptr(), "MctsTreeManager"));
      if (previous_manager)
      {
        Node *old_root = previous_manager->get_root();
        if (old_root && last_action != -1)
        {
          std::unique_ptr<Node> reused_subtree_root = old_root->detach_child(last_action);
          if (reused_subtree_root)
          {
            reused_subtree_root->set_parent(nullptr);
            reused_subtree_root->update_state(current_root_state_py); // Set state for the new root
            tree_manager_ptr = std::make_unique<MctsTreeManager>(std::move(reused_subtree_root));
            root_node = tree_manager_ptr->get_root();
            reused_tree = true;
          }
          else
          {
            std::cerr << "Warning: Could not find child for last_action " << last_action << ". Creating new tree." << std::endl;
          }
        }
        else
        {
          std::cerr << "Warning: Previous tree handle invalid or last_action=-1. Creating new tree." << std::endl;
        }
      }
      else
      {
        std::cerr << "Warning: Failed to retrieve MctsTreeManager pointer. Creating new tree." << std::endl;
      }
    }

    // --- Create New Tree if Reuse Failed ---
    if (!reused_tree)
    {
      try
      {
        // Create root node without state initially
        auto initial_root = std::make_unique<Node>(nullptr, -1, 1.0f); // Parent=null, action=-1, prior=1?
        initial_root->set_root_state(current_root_state_py);           // Set its state
        tree_manager_ptr = std::make_unique<MctsTreeManager>(std::move(initial_root));
        root_node = tree_manager_ptr->get_root();
      }
      catch (const std::exception &e)
      {
        std::cerr << "Error creating initial root node: " << e.what() << std::endl;
        return {{}, py::capsule(nullptr, "MctsTreeManager", capsule_destructor), 0.0};
      }
      catch (const py::error_already_set &e)
      {
        std::cerr << "Python error creating initial root node: " << e.what() << std::endl;
        return {{}, py::capsule(nullptr, "MctsTreeManager", capsule_destructor), 0.0};
      }
    }

    // --- MCTS Core Logic ---
    if (!root_node || root_node->is_terminal())
    { // is_terminal ensures state is materialized
      MctsTreeManager *manager_to_delete = tree_manager_ptr ? tree_manager_ptr.release() : nullptr;
      return {{}, py::capsule(manager_to_delete, "MctsTreeManager", capsule_destructor), 0.0};
    }

    std::mt19937 rng(std::random_device{}());

    // --- Root Preparation ---
    if (!root_node->is_expanded())
    {
      std::vector<Node *> root_batch_nodes = {root_node};
      // Get state for evaluation (materializes if needed)
      std::vector<py::object> root_batch_states;
      try
      {
        root_batch_states.push_back(root_node->get_state());
      }
      catch (...)
      {
        std::cerr << "Error getting root state for initial evaluation." << std::endl;
        MctsTreeManager *manager_raw_ptr = tree_manager_ptr.release();
        return {{}, py::capsule(manager_raw_ptr, "MctsTreeManager", capsule_destructor), 0.0};
      }

      std::vector<NetworkOutput> root_results;
      try
      {
        root_results = trimcts::evaluate_batch_alpha(network_interface_py, root_batch_states);
        if (root_results.empty())
          throw std::runtime_error("Root evaluation returned empty results.");

        if (!root_node->is_terminal())
        {                                            // Checks state again
          root_node->expand(root_results[0].policy); // expand uses get_state internally
          if (!root_node->is_expanded())
          {
            std::cerr << "Warning: Root node failed to expand despite not being terminal." << std::endl;
            MctsTreeManager *manager_raw_ptr = tree_manager_ptr.release();
            return {{}, py::capsule(manager_raw_ptr, "MctsTreeManager", capsule_destructor), 0.0};
          }
        }
        root_node->backpropagate(root_results[0].value);
      }
      catch (const std::exception &e)
      {
        std::cerr << "Error during MCTS root preparation: " << e.what() << std::endl;
        MctsTreeManager *manager_raw_ptr = tree_manager_ptr.release();
        return {{}, py::capsule(manager_raw_ptr, "MctsTreeManager", capsule_destructor), 0.0};
      }
      catch (const py::error_already_set &e)
      {
        std::cerr << "Python error during MCTS root preparation: " << e.what() << std::endl;
        MctsTreeManager *manager_raw_ptr = tree_manager_ptr.release();
        return {{}, py::capsule(manager_raw_ptr, "MctsTreeManager", capsule_destructor), 0.0};
      }
    }
    if (root_node->is_expanded())
    {
      root_node->add_dirichlet_noise(config, rng);
    }

    // --- Simulation Loop ---
    std::vector<Node *> all_leaves_to_evaluate;
    all_leaves_to_evaluate.reserve(config.max_simulations);
    uint64_t total_depth_sum = 0;        // Track sum of depths reached
    uint32_t actual_simulations_run = 0; // Track number of sims completed

    for (uint32_t i = 0; i < config.max_simulations; ++i)
    {
      Node *current_node = root_node;
      int depth = 0;
      bool simulation_completed = false;

      // 1. Selection
      while (current_node->is_expanded() && !current_node->is_terminal()) // is_terminal ensures state
      {
        Node *selected_child = current_node->select_child(config);
        if (!selected_child)
        {
          std::cerr << "Warning: Selection failed for node V=" << current_node->visit_count_ << ". Stopping sim." << std::endl;
          current_node = nullptr;
          break;
        }
        current_node = selected_child;
        depth++;
        if (depth >= config.max_depth)
          break;
      }

      if (!current_node)
        continue;

      // 2. Expansion / Backpropagation
      if (!current_node->is_expanded() && !current_node->is_terminal() && depth < config.max_depth)
      {
        // Leaf node needs evaluation. Add to list. State will be materialized later.
        all_leaves_to_evaluate.push_back(current_node);
        // We will backpropagate after evaluation
        simulation_completed = true; // Mark sim as reaching a point for backprop
      }
      else
      {
        // Node is terminal, already expanded, or max depth reached.
        // Backpropagate terminal outcome or existing value estimate.
        // Need state for get_outcome if terminal.
        Value value = current_node->is_terminal() ? trimcts::get_outcome(current_node->get_state()) : current_node->get_value_estimate();
        current_node->backpropagate(value);
        simulation_completed = true; // Mark sim as reaching a point for backprop
      }

      // Track depth if simulation completed a path
      if (simulation_completed)
      {
        total_depth_sum += depth;
        actual_simulations_run++;
      }

    } // End simulation loop

    // --- Process ALL Collected Leaves in Batches ---
    if (!all_leaves_to_evaluate.empty())
    {
      size_t num_leaves = all_leaves_to_evaluate.size();
      size_t batch_size = static_cast<size_t>(config.mcts_batch_size);

      for (size_t batch_start = 0; batch_start < num_leaves; batch_start += batch_size)
      {
        size_t batch_end = std::min(batch_start + batch_size, num_leaves);
        std::vector<Node *> current_batch_nodes;
        std::vector<py::object> current_batch_states; // States for the network
        current_batch_nodes.reserve(batch_end - batch_start);
        current_batch_states.reserve(batch_end - batch_start);

        // Materialize states for the batch
        bool state_error = false;
        for (size_t k = batch_start; k < batch_end; ++k)
        {
          Node *leaf = all_leaves_to_evaluate[k];
          try
          {
            current_batch_states.push_back(leaf->get_state()); // Materialize state here
            current_batch_nodes.push_back(leaf);
          }
          catch (...)
          {
            std::cerr << "Error getting state for leaf node during batch creation. Skipping node." << std::endl;
            // Backpropagate 0 for this node?
            leaf->backpropagate(0.0f);
            state_error = true;
          }
        }
        // Skip network call if any state failed to materialize in this batch
        if (state_error || current_batch_nodes.empty())
          continue;

        // Process this batch
        try
        {
          std::vector<NetworkOutput> results = trimcts::evaluate_batch_alpha(network_interface_py, current_batch_states);
          process_evaluated_batch(current_batch_nodes, results); // Expands and backpropagates
        }
        catch (const std::exception &e)
        {
          std::cerr << "Error during MCTS batch evaluation/processing (Batch " << (batch_start / batch_size) << "): " << e.what() << std::endl;
          for (Node *leaf : current_batch_nodes)
            leaf->backpropagate(0.0f);
        }
        catch (const py::error_already_set &e)
        {
          std::cerr << "Python error during MCTS batch evaluation/processing (Batch " << (batch_start / batch_size) << "): " << e.what() << std::endl;
          for (Node *leaf : current_batch_nodes)
            leaf->backpropagate(0.0f);
        }
      }
    }

    // --- Collect Results ---
    VisitMap visit_counts;
    if (root_node)
    {
      for (auto const &[action, child_ptr] : root_node->children_)
      {
        visit_counts[action] = child_ptr->visit_count_;
      }
    }

    // --- Calculate Average Depth ---
    double average_depth = (actual_simulations_run > 0)
                               ? static_cast<double>(total_depth_sum) / actual_simulations_run
                               : 0.0;

    // --- Return results and the new tree handle ---
    MctsTreeManager *manager_raw_ptr = tree_manager_ptr.release();
    py::capsule new_capsule(manager_raw_ptr, "MctsTreeManager", &capsule_destructor);

    return std::make_tuple(visit_counts, new_capsule, average_depth);
  }

} // namespace trimcts
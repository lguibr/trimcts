
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

// Make sure pybind11 headers are included here if PYBIND11_EXPORT needs them
#include <pybind11/pybind11.h>
#include <pybind11/stl.h> // Include stl for pair conversion if needed

namespace trimcts
{

  // --- Node Implementation ---
  Node::Node(py::object state, Node *parent, Action action, float prior)
      : parent_(parent), action_taken_(action), state_(std::move(state)), prior_probability_(prior) {}

  bool Node::is_expanded() const { return !children_.empty(); }
  bool Node::is_terminal() const { return trimcts::is_terminal(state_); }

  float Node::get_value_estimate() const
  {
    if (visit_count_ == 0)
      return 0.0f;
    // Avoid division by zero if visit_count_ somehow becomes zero after check (multithreading?)
    // Though MCTS here is single-threaded, being safe doesn't hurt.
    return visit_count_ > 0 ? static_cast<float>(total_action_value_ / visit_count_) : 0.0f;
  }

  float Node::calculate_puct(const SearchConfig &config) const
  {
    if (!parent_)                                    // Root node or detached node has no parent for PUCT calculation relative to parent
      return std::numeric_limits<float>::infinity(); // Prioritize root selection initially.

    // Standard PUCT calculation
    float q_value = get_value_estimate();
    // Use std::max to avoid sqrt(0) if parent_visit_count is 0 (shouldn't happen after root expansion)
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
    // Handle cases where no child is selected (e.g., all scores are -inf)
    if (best_child == nullptr && !children_.empty())
    {
      // Fallback: return the first child? Or handle error?
      // Returning nullptr seems safer, let caller handle.
      std::cerr << "Warning: select_child failed to find a best child despite having children." << std::endl;
    }
    return best_child;
  }

  void Node::expand(const PolicyMap &policy_map)
  {
    if (is_expanded() || is_terminal())
      return;
    std::vector<Action> valid_actions = trimcts::get_valid_actions(state_);
    if (valid_actions.empty())
      return; // Cannot expand if no valid actions

    for (Action action : valid_actions)
    {
      float prior = 0.0f;
      auto it = policy_map.find(action);
      if (it != policy_map.end())
        prior = it->second;
      else
      {
        // Optionally handle actions valid in state but not in policy map
        // prior = 1e-6f; // Example: Small prior
        // std::cerr << "Warning: Action " << action << " valid in state but not in policy map." << std::endl;
      }

      // Eager state creation (copy + step)
      try
      {
        py::object next_state_py = trimcts::copy_state(state_);
        trimcts::apply_action(next_state_py, action);
        children_[action] = std::make_unique<Node>(std::move(next_state_py), this, action, prior);
      }
      catch (const std::exception &e)
      {
        std::cerr << "Error during state copy/step in Node::expand for action " << action << ": " << e.what() << std::endl;
        // Optionally skip this child or handle error differently
      }
      catch (const py::error_already_set &e)
      {
        std::cerr << "Python error during state copy/step in Node::expand for action " << action << ": " << e.what() << std::endl;
        // Optionally skip this child or handle error differently
      }
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

  // Detaches a child node, transferring ownership to the caller.
  std::unique_ptr<Node> Node::detach_child(Action action)
  {
    auto it = children_.find(action);
    if (it == children_.end())
    {
      return nullptr; // Child not found
    }
    std::unique_ptr<Node> child_ptr = std::move(it->second);
    children_.erase(it);
    return child_ptr; // Transfer ownership
  }

  // Updates the Python state object held by this node.
  void Node::update_state(py::object new_state)
  {
    state_ = std::move(new_state); // py::object handles reference counting
  }

  // Sets the parent pointer of this node.
  void Node::set_parent(Node *new_parent)
  {
    parent_ = new_parent;
  }

  void sample_dirichlet_simple(double alpha, size_t k, std::vector<double> &output, std::mt19937 &rng)
  {
    output.resize(k);
    std::gamma_distribution<double> dist(alpha, 1.0);
    double sum = 0.0;
    for (size_t i = 0; i < k; ++i)
    {
      output[i] = dist(rng);
      // Clamp noise to avoid negative or extremely small values before normalization
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
      // Avoid division by zero if sum is too small
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
      // Ensure prior probability is valid before applying noise
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
    // Re-normalize priors after adding noise
    if (std::abs(total_prior - 1.0) > 1e-6 && total_prior > 1e-9)
    {
      for (auto &[action, child_ptr] : children_)
      {
        child_ptr->prior_probability_ /= static_cast<float>(total_prior);
      }
    }
    else if (total_prior <= 1e-9 && num_children > 0)
    {
      // Handle case where all priors became zero (unlikely but possible)
      float uniform_prior = 1.0f / static_cast<float>(num_children);
      for (auto &[action, child_ptr] : children_)
      {
        child_ptr->prior_probability_ = uniform_prior;
      }
      std::cerr << "Warning: Total prior probability near zero after adding noise. Resetting to uniform." << std::endl;
    }
  }

  // --- MCTS Main Logic with Tree Reuse ---

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
      if (!leaf->is_terminal())
      {
        try
        {
          leaf->expand(output.policy);
        }
        catch (const std::exception &e)
        {
          std::cerr << "Error during leaf expansion: " << e.what() << std::endl;
          // Decide how to handle - maybe just backpropagate value?
        }
        catch (const py::error_already_set &e)
        {
          std::cerr << "Python error during leaf expansion: " << e.what() << std::endl;
        }
      }
      // Backpropagate the value regardless of expansion success/failure
      leaf->backpropagate(output.value);
    }
  }

  // Main MCTS function updated for tree reuse
  PYBIND11_EXPORT std::pair<VisitMap, py::capsule> run_mcts_cpp_internal(
      py::object current_root_state_py,
      py::object network_interface_py,
      const SearchConfig &config,
      const py::capsule &previous_tree_capsule, // Optional handle from previous step
      Action last_action                        // Action that led to current_root_state_py
  )
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
        // Check if old_root is valid and last_action makes sense
        if (old_root && last_action != -1)
        {
          // Attempt to find and detach the child corresponding to the last action
          std::unique_ptr<Node> reused_subtree_root = old_root->detach_child(last_action);

          if (reused_subtree_root)
          {
            // Successfully detached the child subtree
            reused_subtree_root->set_parent(nullptr);                 // It's the new root
            reused_subtree_root->update_state(current_root_state_py); // Update state reference

            // Create a new manager owning the reused subtree
            tree_manager_ptr = std::make_unique<MctsTreeManager>(std::move(reused_subtree_root));
            root_node = tree_manager_ptr->get_root();
            reused_tree = true;
            // std::cout << "[DEBUG] MCTS Tree Reused for action: " << last_action << std::endl;

            // Note: The old manager (and the rest of the old tree) will be deleted
            // when the previous_tree_capsule goes out of scope in Python and its
            // destructor (capsule_destructor) is called.
          }
          else
          {
            std::cerr << "Warning: Could not find child for last_action " << last_action << " in previous tree. Creating new tree." << std::endl;
          }
        }
        else
        {
          std::cerr << "Warning: Previous tree handle provided, but old root or last_action invalid. Creating new tree." << std::endl;
        }
      }
      else
      {
        std::cerr << "Warning: Failed to retrieve MctsTreeManager pointer from capsule. Creating new tree." << std::endl;
      }
    }

    // --- Create New Tree if Reuse Failed or Not Requested ---
    if (!reused_tree)
    {
      // std::cout << "[DEBUG] MCTS Creating New Tree." << std::endl;
      try
      {
        auto initial_root = std::make_unique<Node>(current_root_state_py);
        tree_manager_ptr = std::make_unique<MctsTreeManager>(std::move(initial_root));
        root_node = tree_manager_ptr->get_root();
      }
      catch (const std::exception &e)
      {
        std::cerr << "Error creating initial root node: " << e.what() << std::endl;
        return {{}, py::capsule(nullptr, "MctsTreeManager", capsule_destructor)};
      }
      catch (const py::error_already_set &e)
      {
        std::cerr << "Python error creating initial root node: " << e.what() << std::endl;
        return {{}, py::capsule(nullptr, "MctsTreeManager", capsule_destructor)};
      }
    }

    // --- MCTS Core Logic (largely unchanged, operates on root_node) ---
    if (!root_node || root_node->is_terminal())
    {
      // Return empty results and a null capsule if root is invalid or terminal
      // If tree_manager_ptr exists, release it to the capsule so it gets deleted.
      MctsTreeManager *manager_to_delete = tree_manager_ptr ? tree_manager_ptr.release() : nullptr;
      return {{}, py::capsule(manager_to_delete, "MctsTreeManager", capsule_destructor)};
    }

    std::mt19937 rng(std::random_device{}());

    // --- Root Preparation (Expansion/Noise) ---
    // Expand if the node is not expanded. Add noise if it's expanded (either newly or reused).
    if (!root_node->is_expanded())
    {
      std::vector<Node *> root_batch_nodes = {root_node};
      std::vector<py::object> root_batch_states = {root_node->state_};
      std::vector<NetworkOutput> root_results;
      try
      {
        root_results = trimcts::evaluate_batch_alpha(network_interface_py, root_batch_states);
        if (root_results.empty())
        {
          throw std::runtime_error("Root evaluation returned empty results.");
        }
        if (!root_node->is_terminal())
        {
          root_node->expand(root_results[0].policy);
          if (!root_node->is_expanded())
          {
            std::cerr << "Warning: Root node failed to expand despite not being terminal." << std::endl;
            // Return empty results, but keep the tree state
            MctsTreeManager *manager_raw_ptr = tree_manager_ptr.release();
            return {{}, py::capsule(manager_raw_ptr, "MctsTreeManager", capsule_destructor)};
          }
        }
        // Backpropagate the root's evaluated value *once* after initial eval/expansion
        root_node->backpropagate(root_results[0].value);
      }
      catch (const std::exception &e)
      {
        std::cerr << "Error during MCTS root preparation: " << e.what() << std::endl;
        MctsTreeManager *manager_raw_ptr = tree_manager_ptr.release();
        return {{}, py::capsule(manager_raw_ptr, "MctsTreeManager", capsule_destructor)};
      }
      catch (const py::error_already_set &e)
      {
        std::cerr << "Python error during MCTS root preparation: " << e.what() << std::endl;
        MctsTreeManager *manager_raw_ptr = tree_manager_ptr.release();
        return {{}, py::capsule(manager_raw_ptr, "MctsTreeManager", capsule_destructor)};
      }
    }
    // Add noise if the root is now expanded (either newly or reused)
    if (root_node->is_expanded())
    {
      root_node->add_dirichlet_noise(config, rng);
    }

    // --- Simulation Loop (Batching logic remains the same) ---
    std::vector<Node *> all_leaves_to_evaluate;
    all_leaves_to_evaluate.reserve(config.max_simulations); // Reserve based on sims

    for (uint32_t i = 0; i < config.max_simulations; ++i)
    {
      Node *current_node = root_node; // Start from the potentially reused root
      int depth = 0;

      // 1. Selection
      while (current_node->is_expanded() && !current_node->is_terminal())
      {
        Node *selected_child = current_node->select_child(config);
        if (!selected_child)
        {
          std::cerr << "Warning: Selection failed to find a child for node with visit count " << current_node->visit_count_ << ". Stopping simulation." << std::endl;
          current_node = nullptr; // Mark selection as failed
          break;
        }
        current_node = selected_child;
        depth++;
        if (depth >= config.max_depth)
          break;
      }

      if (!current_node)
        continue; // Skip to next simulation if selection failed

      // 2. Check if Expansion is Needed (or if terminal/max depth)
      if (!current_node->is_expanded() && !current_node->is_terminal() && depth < config.max_depth)
      {
        all_leaves_to_evaluate.push_back(current_node);
      }
      else
      {
        // Backpropagate terminal outcome or existing value estimate
        Value value = current_node->is_terminal() ? trimcts::get_outcome(current_node->state_) : current_node->get_value_estimate();
        current_node->backpropagate(value);
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
        std::vector<py::object> current_batch_states;
        current_batch_nodes.reserve(batch_end - batch_start);
        current_batch_states.reserve(batch_end - batch_start);

        for (size_t k = batch_start; k < batch_end; ++k)
        {
          current_batch_nodes.push_back(all_leaves_to_evaluate[k]);
          current_batch_states.push_back(all_leaves_to_evaluate[k]->state_);
        }

        try
        {
          std::vector<NetworkOutput> results = trimcts::evaluate_batch_alpha(network_interface_py, current_batch_states);
          process_evaluated_batch(current_batch_nodes, results);
        }
        catch (const std::exception &e)
        {
          std::cerr << "Error during MCTS batch evaluation/processing (Batch " << (batch_start / batch_size) << "): " << e.what() << std::endl;
          for (Node *leaf : current_batch_nodes)
          {
            leaf->backpropagate(0.0f); // Backpropagate neutral value on error
          }
        }
        catch (const py::error_already_set &e)
        {
          std::cerr << "Python error during MCTS batch evaluation/processing (Batch " << (batch_start / batch_size) << "): " << e.what() << std::endl;
          for (Node *leaf : current_batch_nodes)
          {
            leaf->backpropagate(0.0f);
          }
        }
      }
    }

    // --- Collect Results ---
    VisitMap visit_counts;
    if (root_node)
    { // Ensure root_node is valid before accessing children
      for (auto const &[action, child_ptr] : root_node->children_)
      {
        visit_counts[action] = child_ptr->visit_count_;
      }
    }

    // --- Return results and the new tree handle ---
    // Release ownership from unique_ptr to pass raw pointer to capsule
    MctsTreeManager *manager_raw_ptr = tree_manager_ptr.release();
    // Create the capsule with the pointer and the C destructor function
    py::capsule new_capsule(manager_raw_ptr, "MctsTreeManager", &capsule_destructor);

    return {visit_counts, new_capsule};
  }

} // namespace trimcts
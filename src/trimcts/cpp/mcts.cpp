// File: src/trimcts/cpp/mcts.cpp
#include "mcts.h"
#include "python_interface.h" // For Python interaction
#include <cmath>
#include <limits>
#include <stdexcept>
#include <iostream> // For temporary debugging
#include <numeric>  // For std::accumulate
#include <vector>
#include <algorithm> // For std::max_element, std::max, std::min
#include <chrono>    // For timing (optional debug)

// Make sure pybind11 headers are included here if PYBIND11_EXPORT needs them
#include <pybind11/pybind11.h>

namespace trimcts
{

  // --- Node Implementation (No changes needed here) ---
  Node::Node(py::object state, Node *parent, Action action, float prior)
      : parent_(parent), action_taken_(action), state_(std::move(state)), prior_probability_(prior) {}

  bool Node::is_expanded() const { return !children_.empty(); }
  bool Node::is_terminal() const { return trimcts::is_terminal(state_); }

  float Node::get_value_estimate() const
  {
    if (visit_count_ == 0)
      return 0.0f;
    return static_cast<float>(total_action_value_ / visit_count_);
  }

  float Node::calculate_puct(const SearchConfig &config) const
  {
    if (!parent_)
      return -std::numeric_limits<float>::infinity();
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
    return best_child;
  }

  void Node::expand(const PolicyMap &policy_map)
  {
    if (is_expanded() || is_terminal())
      return;
    std::vector<Action> valid_actions = trimcts::get_valid_actions(state_);
    if (valid_actions.empty())
      return;

    for (Action action : valid_actions)
    {
      float prior = 0.0f;
      auto it = policy_map.find(action);
      if (it != policy_map.end())
        prior = it->second;
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
      for (size_t i = 0; i < k; ++i)
        output[i] = 1.0 / k;
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
      child_ptr->prior_probability_ = (1.0f - config.dirichlet_epsilon) * child_ptr->prior_probability_ + config.dirichlet_epsilon * static_cast<float>(noise[i]);
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
  }

  // --- MCTS Main Logic with Corrected Batching ---

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
        leaf->expand(output.policy);
      leaf->backpropagate(output.value);
    }
  }

  PYBIND11_EXPORT VisitMap run_mcts_cpp_internal(
      py::object root_state_py,
      py::object network_interface_py,
      const SearchConfig &config)
  {
    if (trimcts::is_terminal(root_state_py))
      return {};

    Node root(std::move(root_state_py));
    std::mt19937 rng(std::random_device{}());

    // --- Root Preparation ---
    std::vector<Node *> root_batch_nodes = {&root};
    std::vector<py::object> root_batch_states = {root.state_};
    std::vector<NetworkOutput> root_results;
    try
    {
      root_results = trimcts::evaluate_batch_alpha(network_interface_py, root_batch_states);
      if (root_results.empty())
        throw std::runtime_error("Root evaluation returned empty results.");
      if (!root.is_terminal())
      {
        root.expand(root_results[0].policy);
        if (root.is_expanded())
          root.add_dirichlet_noise(config, rng);
        else
        {
          std::cerr << "Warning: Root node failed to expand despite not being terminal." << std::endl;
          return {};
        }
      }
      // Backpropagate root value once before simulations
      root.backpropagate(root_results[0].value);
    }
    catch (const std::exception &e)
    {
      std::cerr << "Error during MCTS root initialization/evaluation: " << e.what() << std::endl;
      return {};
    }

    // --- Simulation Loop ---
    std::vector<Node *> all_leaves_to_evaluate; // Collect ALL leaves here

    for (uint32_t i = 0; i < config.max_simulations; ++i)
    {
      Node *current_node = &root;
      int depth = 0;

      // 1. Selection
      while (current_node->is_expanded() && !current_node->is_terminal())
      {
        Node *selected_child = current_node->select_child(config);
        if (!selected_child)
        {
          // This might happen if all children have invalid PUCT scores
          // Or if the node was expanded but somehow has no children (logic error).
          std::cerr << "Warning: Selection failed to find a child for node with visit count " << current_node->visit_count_ << ". Stopping simulation." << std::endl;
          current_node = nullptr; // Mark selection as failed
          break;                  // Exit selection loop for this simulation
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
        // Leaf node needs evaluation and expansion. Add to collection.
        all_leaves_to_evaluate.push_back(current_node); // Add to the main list
      }
      else
      {
        // Node is terminal, already expanded, or max depth reached.
        // Backpropagate the existing value estimate or terminal outcome immediately.
        Value value = current_node->is_terminal() ? trimcts::get_outcome(current_node->state_) : current_node->get_value_estimate();
        current_node->backpropagate(value);
      }

    } // End simulation loop

    // --- Process ALL Collected Leaves in Batches ---
    if (!all_leaves_to_evaluate.empty())
    {
      size_t num_leaves = all_leaves_to_evaluate.size();
      size_t batch_size = static_cast<size_t>(config.mcts_batch_size); // Cast once

      for (size_t batch_start = 0; batch_start < num_leaves; batch_start += batch_size)
      {
        size_t batch_end = std::min(batch_start + batch_size, num_leaves);
        std::vector<Node *> current_batch_nodes;
        std::vector<py::object> current_batch_states;
        current_batch_nodes.reserve(batch_end - batch_start);
        current_batch_states.reserve(batch_end - batch_start);

        // Create sub-vectors for the current batch
        for (size_t k = batch_start; k < batch_end; ++k)
        {
          current_batch_nodes.push_back(all_leaves_to_evaluate[k]);
          current_batch_states.push_back(all_leaves_to_evaluate[k]->state_);
        }

        // Process this batch
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
            leaf->backpropagate(0.0f);
          }
        }
      }
    }

    // --- Collect Results ---
    VisitMap visit_counts;
    for (auto const &[action, child_ptr] : root.children_)
    {
      visit_counts[action] = child_ptr->visit_count_;
    }

    return visit_counts;
  }

} // namespace trimcts
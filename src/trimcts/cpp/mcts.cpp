
#include "mcts.h"
#include "python_interface.h" // For Python interaction
#include <cmath>
#include <limits>
#include <stdexcept>
#include <iostream> // For temporary debugging
#include <numeric>  // For std::accumulate
#include <vector>
#include <algorithm> // For std::max_element

namespace trimcts
{

  // --- Node Implementation ---

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
    // PUCT formula: Q(s,a) + C_puct * P(s,a) * sqrt(N(s)) / (1 + N(s,a))
    // Here, 'this' node represents the child (s,a)
    // Q(s,a) is the value estimate of this child node
    // P(s,a) is the prior probability of this child node
    // N(s) is the visit count of the parent node
    // N(s,a) is the visit count of this child node

    if (!parent_)
    {
      // Should not happen for child selection, but handle defensively
      return -std::numeric_limits<float>::infinity();
    }

    float q_value = get_value_estimate();
    double parent_visits_sqrt = std::sqrt(static_cast<double>(std::max(1, parent_->visit_count_)));
    double exploration_term = config.cpuct * prior_probability_ * (parent_visits_sqrt / (1.0 + visit_count_));

    return q_value + static_cast<float>(exploration_term);
  }

  Node *Node::select_child(const SearchConfig &config)
  {
    if (!is_expanded())
    {
      return nullptr; // Cannot select child if not expanded
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
    return best_child; // Can be nullptr if children_ is empty (shouldn't happen if is_expanded is true)
  }

  void Node::expand(const PolicyMap &policy_map)
  {
    if (is_expanded() || is_terminal())
    {
      return; // Don't expand if already expanded or terminal
    }

    std::vector<Action> valid_actions = trimcts::get_valid_actions(state_);
    if (valid_actions.empty())
    {
      // Handle case where Python state says not terminal but has no actions
      // This might indicate an issue in the Python state logic or MCTS reached a dead end
      // For now, just don't expand. Consider logging a warning.
      std::cerr << "Warning: Node::expand called on non-terminal state with no valid actions." << std::endl;
      return;
    }

    for (Action action : valid_actions)
    {
      float prior = 0.0f;
      auto it = policy_map.find(action);
      if (it != policy_map.end())
      {
        prior = it->second;
      }

      // Create next state by copying and applying action (calls Python)
      py::object next_state_py = trimcts::copy_state(state_);
      trimcts::apply_action(next_state_py, action); // Modifies next_state_py in-place

      // Create the child node
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

  // Simple gamma distribution for Dirichlet noise (replace with proper library if needed)
  // This is a placeholder and likely not statistically correct for true Dirichlet.
  void sample_dirichlet_simple(double alpha, size_t k, std::vector<double> &output, std::mt19937 &rng)
  {
    output.resize(k);
    std::gamma_distribution<double> dist(alpha, 1.0);
    double sum = 0.0;
    for (size_t i = 0; i < k; ++i)
    {
      output[i] = dist(rng);
      if (output[i] < 1e-9)
        output[i] = 1e-9; // Avoid zero
      sum += output[i];
    }
    if (sum > 1e-9)
    {
      for (size_t i = 0; i < k; ++i)
      {
        output[i] /= sum;
      }
    }
    else
    { // Handle sum near zero case (e.g., all samples were tiny)
      for (size_t i = 0; i < k; ++i)
      {
        output[i] = 1.0 / k;
      }
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
    sample_dirichlet_simple(config.dirichlet_alpha, num_children, noise, rng); // Use simple placeholder

    size_t i = 0;
    double total_prior = 0.0;
    for (auto &[action, child_ptr] : children_)
    {
      child_ptr->prior_probability_ = (1.0f - config.dirichlet_epsilon) * child_ptr->prior_probability_ + config.dirichlet_epsilon * static_cast<float>(noise[i]);
      total_prior += child_ptr->prior_probability_;
      i++;
    }

    // Re-normalize (optional, but good practice if noise addition causes drift)
    if (std::abs(total_prior - 1.0) > 1e-6 && total_prior > 1e-9)
    {
      for (auto &[action, child_ptr] : children_)
      {
        child_ptr->prior_probability_ /= static_cast<float>(total_prior);
      }
    }
  }

  // --- MCTS Main Logic ---

  VisitMap run_mcts_cpp_internal(
      py::object root_state_py,
      py::object network_interface_py, // AlphaZero interface for now
      const SearchConfig &config)
  {
    // Basic check: Ensure root state is not terminal
    if (trimcts::is_terminal(root_state_py))
    {
      std::cerr << "Error: MCTS called on a terminal root state." << std::endl;
      return {}; // Return empty map
    }

    // Create root node
    Node root(std::move(root_state_py));
    std::mt19937 rng(std::random_device{}()); // Random number generator for noise

    // Initial evaluation and expansion of the root
    try
    {
      NetworkOutput root_eval = trimcts::evaluate_state_alpha(network_interface_py, root.state_);
      root.expand(root_eval.policy);
      if (root.is_expanded())
      {
        root.add_dirichlet_noise(config, rng);
      }
      else if (!root.is_terminal())
      {
        std::cerr << "Warning: Root node failed to expand." << std::endl;
        // If root didn't expand but isn't terminal, MCTS can't proceed meaningfully
        return {};
      }
      // Backpropagate initial root value estimate (optional, depends on exact AlphaZero variant)
      // root.backpropagate(root_eval.value); // Let's skip this for now, value comes from simulations
    }
    catch (const std::exception &e)
    {
      std::cerr << "Error during MCTS root initialization: " << e.what() << std::endl;
      return {};
    }

    for (uint32_t i = 0; i < config.max_simulations; ++i)
    {
      Node *current_node = &root;
      std::vector<Node *> path;
      path.push_back(current_node);
      int depth = 0;

      // 1. Selection
      while (current_node->is_expanded() && !current_node->is_terminal())
      {
        current_node = current_node->select_child(config);
        if (!current_node)
        {
          std::cerr << "Error: Selection returned nullptr." << std::endl;
          goto next_simulation; // Skip to next simulation on error
        }
        path.push_back(current_node);
        depth++;
        if (depth >= config.max_depth)
          break; // Stop if max depth reached
      }

      // 2. Expansion & Evaluation
      Value value;
      if (!current_node->is_terminal() && depth < config.max_depth)
      {
        try
        {
          // Evaluate the leaf node
          NetworkOutput leaf_eval = trimcts::evaluate_state_alpha(network_interface_py, current_node->state_);
          value = leaf_eval.value;
          // Expand the leaf node
          current_node->expand(leaf_eval.policy);
        }
        catch (const std::exception &e)
        {
          std::cerr << "Error during MCTS expansion/evaluation: " << e.what() << std::endl;
          // Decide how to handle evaluation errors, e.g., backpropagate 0 or skip
          value = 0.0; // Backpropagate neutral value on error
        }
      }
      else
      {
        // If terminal or max depth reached, use the terminal outcome or current estimate
        value = current_node->is_terminal() ? trimcts::get_outcome(current_node->state_) : current_node->get_value_estimate();
      }

      // 3. Backpropagation
      current_node->backpropagate(value); // Backpropagate from the leaf (or max depth node)

    next_simulation:; // Label for goto
    }

    // Collect visit counts from root's children
    VisitMap visit_counts;
    for (auto const &[action, child_ptr] : root.children_)
    {
      visit_counts[action] = child_ptr->visit_count_;
    }

    return visit_counts;
  }

} // namespace trimcts
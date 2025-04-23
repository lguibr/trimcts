#pragma once

#include <pybind11/pybind11.h> // Include pybind11 first
#include <vector>
#include <map>
#include <memory> // For std::unique_ptr
#include <random>
#include <optional> // For optional members/return
#include <utility>  // For std::pair
#include <tuple>    // For std::tuple

#include "config.h"
#include "python_interface.h" // For types and Python interaction helpers

namespace py = pybind11;

namespace trimcts
{

  // Forward declaration
  class MctsTreeManager;

  class Node
  {
  public:
    // Constructor for non-root nodes
    Node(Node *parent, Action action, float prior = 0.0);
    ~Node() = default; // Use default destructor

    // Disable copy constructor and assignment operator
    Node(const Node &) = delete;
    Node &operator=(const Node &) = delete;

    // Enable move constructor and assignment operator
    Node(Node &&) = default;
    Node &operator=(Node &&) = default;

    // --- Core MCTS Methods ---
    bool is_expanded() const;
    bool is_terminal(); // Now non-const as it might materialize state
    float get_value_estimate() const;
    Node *select_child(const SearchConfig &config);
    void expand(const PolicyMap &policy_map); // Expansion logic changes
    void backpropagate(float value);
    void add_dirichlet_noise(const SearchConfig &config, std::mt19937 &rng);

    // --- State Management (Lazy Creation) ---
    py::object get_state(); // Gets or materializes the state
    bool has_state_materialized() const;
    void set_root_state(py::object state);   // Special method for root
    void update_state(py::object new_state); // ADDED declaration back

    // --- Tree Structure ---
    std::unique_ptr<Node> detach_child(Action action);
    void set_parent(Node *new_parent);
    Node *get_parent() const;
    Action get_action_taken() const; // Action that led *to* this node

    // --- Public Members ---
    // Node *parent_; // Keep parent pointer
    // Action action_taken_; // Renamed to action_to_reach_? No, keep as action that *led* here.
    std::map<Action, std::unique_ptr<Node>> children_;

    int visit_count_ = 0;
    double total_action_value_ = 0.0; // Use double for accumulation
    float prior_probability_ = 0.0;

  private:
    Node *parent_;        // Pointer to parent node
    Action action_taken_; // Action taken by parent to reach this node

    // State is now optional and mutable for lazy creation
    mutable std::optional<py::object> materialized_state_;
    mutable bool state_materialized_ = false;

    float calculate_puct(const SearchConfig &config) const;
  };

  // Main MCTS function signature updated to return average depth
  PYBIND11_EXPORT std::tuple<VisitMap, py::capsule, double> run_mcts_cpp_internal(
      py::object current_root_state_py,
      py::object network_interface_py,
      const SearchConfig &config,
      const py::capsule &previous_tree_capsule,
      Action last_action);

} // namespace trimcts
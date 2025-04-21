#pragma once

#include <pybind11/pybind11.h> // Include pybind11 first
#include <vector>
#include <map>
#include <memory> // For std::unique_ptr
#include <random>
#include <optional> // For optional return
#include <utility>  // For std::pair

#include "config.h"
#include "python_interface.h" // For types and Python interaction helpers
// #include "structs.h"       // REMOVED redundant include

namespace py = pybind11;

namespace trimcts
{

  // Forward declaration
  class MctsTreeManager;

  class Node
  {
  public:
    // Constructor now takes py::object state
    Node(py::object state, Node *parent = nullptr, Action action = -1, float prior = 0.0);
    ~Node() = default; // Use default destructor

    // Disable copy constructor and assignment operator
    Node(const Node &) = delete;
    Node &operator=(const Node &) = delete;

    // Enable move constructor and assignment operator
    Node(Node &&) = default;
    Node &operator=(Node &&) = default;

    bool is_expanded() const;
    bool is_terminal() const;
    float get_value_estimate() const;
    Node *select_child(const SearchConfig &config);
    void expand(const PolicyMap &policy_map);
    void backpropagate(float value);
    void add_dirichlet_noise(const SearchConfig &config, std::mt19937 &rng);

    // Method to detach a child node (transfers ownership)
    std::unique_ptr<Node> detach_child(Action action);

    // Method to update the state object (needed for reuse)
    void update_state(py::object new_state);

    // Method to set parent (needed for reuse)
    void set_parent(Node *new_parent);

    // --- Public Members ---
    Node *parent_;
    Action action_taken_; // Action that led to this node
    py::object state_;    // Python GameState object (KEEPING FOR NOW)
    std::map<Action, std::unique_ptr<Node>> children_;

    int visit_count_ = 0;
    double total_action_value_ = 0.0; // Use double for accumulation
    float prior_probability_ = 0.0;

  private:
    float calculate_puct(const SearchConfig &config) const;
  };

  // Main MCTS function signature updated for tree reuse
  // Returns VisitMap and the new tree handle (capsule)
  PYBIND11_EXPORT std::pair<VisitMap, py::capsule> run_mcts_cpp_internal(
      py::object current_root_state_py,
      py::object network_interface_py,
      const SearchConfig &config,
      const py::capsule &previous_tree_capsule, // Optional handle from previous step
      Action last_action                        // Action that led to current_root_state_py
  );

} // namespace trimcts
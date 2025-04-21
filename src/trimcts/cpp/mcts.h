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

  // Main MCTS function signature with visibility macro
  PYBIND11_EXPORT VisitMap run_mcts_cpp_internal( // Added PYBIND11_EXPORT
      py::object root_state,
      py::object network_interface, // AlphaZero interface for now
      const SearchConfig &config);

} // namespace trimcts
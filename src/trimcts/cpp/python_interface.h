// File: src/trimcts/cpp/python_interface.h
#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h> // For automatic vector/map conversions
#include <vector>
#include <map>       // Added for std::map
#include <stdexcept> // For std::runtime_error
#include <string>

namespace py = pybind11;

namespace trimcts
{

  // Define basic types used across C++/Python
  using Action = int;
  using Value = float;
  using PolicyMap = std::map<Action, float>;
  using VisitMap = std::map<Action, int>; // Fixed alias

  // Helper struct to hold evaluation results from Python network
  struct NetworkOutput
  {
    PolicyMap policy;
    Value value;
  };

  // --- Helper functions to interact with Python objects ---

  inline py::object call_python_method(py::handle obj, const char *method_name)
  {
    try
    {
      return obj.attr(method_name)();
    }
    catch (py::error_already_set &e)
    {
      throw std::runtime_error("Python error in method '" + std::string(method_name) + "': " + e.what());
    }
    catch (const std::exception &e)
    {
      throw std::runtime_error("C++ error calling method '" + std::string(method_name) + "': " + e.what());
    }
  }

  template <typename Arg>
  inline py::object call_python_method(py::handle obj, const char *method_name, Arg &&arg)
  {
    try
    {
      return obj.attr(method_name)(std::forward<Arg>(arg));
    }
    catch (py::error_already_set &e)
    {
      throw std::runtime_error("Python error in method '" + std::string(method_name) + "': " + e.what());
    }
    catch (const std::exception &e)
    {
      throw std::runtime_error("C++ error calling method '" + std::string(method_name) + "': " + e.what());
    }
  }

  // --- Game State Interface ---
  // These functions call methods on the Python GameState object

  inline py::object copy_state(py::handle py_state)
  {
    return call_python_method(py_state, "copy");
  }

  inline bool is_terminal(py::handle py_state)
  {
    return call_python_method(py_state, "is_over").cast<bool>();
  }

  inline Value get_outcome(py::handle py_state)
  {
    // AlphaZero expects outcome only for terminal states
    if (!is_terminal(py_state))
    {
      return 0.0f; // Return 0 for non-terminal states
    }
    return call_python_method(py_state, "get_outcome").cast<Value>();
  }

  inline std::vector<Action> get_valid_actions(py::handle py_state)
  {
    py::object result = call_python_method(py_state, "valid_actions");
    try
    {
      return result.cast<std::vector<Action>>();
    }
    catch (const py::cast_error &)
    {
      try
      {
        py::set py_set = result.cast<py::set>();
        std::vector<Action> actions;
        actions.reserve(py_set.size());
        for (py::handle item : py_set)
        {
          actions.push_back(item.cast<Action>());
        }
        return actions;
      }
      catch (const py::cast_error &)
      {
        throw std::runtime_error("Python 'valid_actions' must return list or set of int.");
      }
    }
  }

  // Apply the action in-place; no return value needed
  inline void apply_action(py::handle py_state, Action action)
  {
    call_python_method(py_state, "step", action);
  }

  // --- Network Interface for AlphaZero ---

  inline NetworkOutput evaluate_state_alpha(py::handle py_network, py::handle py_state)
  {
    py::tuple result = call_python_method(py_network, "evaluate_state", py_state).cast<py::tuple>();
    if (result.size() != 2)
      throw std::runtime_error("Python 'evaluate_state' must return (policy_dict, value).");
    PolicyMap policy = result[0].cast<PolicyMap>();
    Value value = result[1].cast<Value>();
    return {policy, value};
  }

  inline std::vector<NetworkOutput> evaluate_batch_alpha(
      py::handle py_network,
      const std::vector<py::object> &py_states)
  {
    py::list state_list = py::cast(py_states);
    py::list results_list = call_python_method(py_network, "evaluate_batch", state_list).cast<py::list>();

    if (results_list.size() != py_states.size())
      throw std::runtime_error("Python 'evaluate_batch' returned wrong length.");

    std::vector<NetworkOutput> outputs;
    outputs.reserve(py_states.size());
    for (auto item : results_list)
    {
      py::tuple tup = item.cast<py::tuple>();
      if (tup.size() != 2)
        throw std::runtime_error("Each 'evaluate_batch' item must be (policy_dict, value).");
      outputs.push_back({tup[0].cast<PolicyMap>(), tup[1].cast<Value>()});
    }
    return outputs;
  }

  // (MuZero interfaces to be added later)

} // namespace trimcts

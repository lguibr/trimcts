// File: trimcts/src/trimcts/cpp/bindings.cpp
// File: src/trimcts/cpp/bindings.cpp

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>     // For map/vector/pair conversions
#include <pybind11/pytypes.h> // For py::object, py::handle, py::capsule

#include "mcts.h"             // C++ MCTS logic header (includes Node)
#include "mcts_manager.h"     // Include the manager header for capsule name/destructor
#include "config.h"           // C++ SearchConfig struct
#include "python_interface.h" // For NetworkOutput, VisitMap, etc.
// #include "structs.h"       // REMOVED redundant include
#include <string>    // std::string
#include <stdexcept> // std::runtime_error
#include <utility>   // For std::pair
#include <tuple>     // For std::tuple

namespace py = pybind11;
namespace tc = trimcts; // Alias for your C++ namespace

// Helper function to transfer config from Python Pydantic model to C++ struct
static tc::SearchConfig python_to_cpp_config(const py::object &py_config)
{
  tc::SearchConfig cpp_config;
  try
  {
    cpp_config.max_simulations = py_config.attr("max_simulations").cast<uint32_t>();
    cpp_config.max_depth = py_config.attr("max_depth").cast<uint32_t>();
    cpp_config.cpuct = py_config.attr("cpuct").cast<double>();
    cpp_config.dirichlet_alpha = py_config.attr("dirichlet_alpha").cast<double>();
    cpp_config.dirichlet_epsilon = py_config.attr("dirichlet_epsilon").cast<double>();
    cpp_config.discount = py_config.attr("discount").cast<double>();
    cpp_config.mcts_batch_size = py_config.attr("mcts_batch_size").cast<uint32_t>();
  }
  catch (const py::error_already_set &e)
  {
    throw std::runtime_error(
        std::string("Error accessing SearchConfiguration attributes: ") + e.what());
  }
  catch (const std::exception &e)
  {
    throw std::runtime_error(
        std::string("Error converting SearchConfiguration: ") + e.what());
  }
  return cpp_config;
}

// Wrapper function exposed to Python - updated for tree reuse
// Returns a tuple (VisitMap, new_capsule, avg_depth)
std::tuple<tc::VisitMap, py::capsule, double> run_mcts_cpp_wrapper(
    py::object root_state_py,
    py::object network_interface_py,
    const py::object &config_py,
    const py::object &previous_tree_handle_obj, // Accept py::object (can be None)
    tc::Action last_action                      // Action leading to root_state_py
)
{
  tc::SearchConfig config_cpp = python_to_cpp_config(config_py);

  // Handle optional capsule input
  py::capsule previous_tree_capsule;
  if (!previous_tree_handle_obj.is_none())
  {
    try
    {
      // Attempt to cast the Python object to a capsule
      previous_tree_capsule = previous_tree_handle_obj.cast<py::capsule>();
      // Optional: Check capsule name for extra safety, though PyCapsule_IsValid does this
      // if (!PyCapsule_IsValid(previous_tree_capsule.ptr(), "MctsTreeManager")) {
      //     throw py::type_error("Invalid capsule type passed for previous_tree_handle.");
      // }
    }
    catch (const py::cast_error &e)
    {
      throw py::type_error("Argument previous_tree_handle must be a valid capsule or None.");
    }
  }
  // If previous_tree_handle_obj was None, previous_tree_capsule remains default-constructed (invalid/null)

  try
  {
    // Call the internal C++ MCTS implementation with reuse parameters
    return tc::run_mcts_cpp_internal(
        root_state_py,
        network_interface_py,
        config_cpp,
        previous_tree_capsule, // Pass the potentially null capsule
        last_action);
  }
  catch (const std::exception &e)
  {
    throw py::value_error(std::string("Error in C++ MCTS execution: ") + e.what());
  }
  catch (const py::error_already_set &)
  {
    throw; // Re-throw Python exceptions
  }
}

PYBIND11_MODULE(trimcts_cpp, m)
{
  m.doc() = "C++ core module for TriMCTS with Tree Reuse";

  // Expose the updated MCTS function
  m.def("run_mcts_cpp",
        &run_mcts_cpp_wrapper,
        py::arg("root_state"),
        py::arg("network_interface"),
        py::arg("config"),
        py::arg("previous_tree_handle"), // Use a descriptive name
        py::arg("last_action"),
        R"pbdoc(
            Runs MCTS simulations from the root state using the provided network interface and configuration.
            Supports tree reuse via previous_tree_handle and last_action.

            Args:
                root_state: The current Python game state object.
                network_interface: Python object implementing the network evaluation interface.
                config: Python SearchConfiguration object.
                previous_tree_handle: An optional handle (py::capsule or None) to the MCTS tree state from the previous step.
                last_action: The action taken in the previous step that led to the current root_state. Use -1 if no previous step or handle.

            Returns:
                A tuple containing:
                    - VisitMap (dict[int, int]): Visit counts for actions from the root.
                    - py::capsule: A new handle to the MCTS tree state after the search. None if MCTS failed or state was terminal.
                    - float: The average depth reached across all simulations performed.
        )pbdoc",
        // Ensure return value policy allows Python to manage the capsule lifetime
        py::return_value_policy::move); // Move the tuple, including the capsule

#ifdef VERSION_INFO
  m.attr("__version__") = VERSION_INFO;
#else
  m.attr("__version__") = "dev";
#endif
}
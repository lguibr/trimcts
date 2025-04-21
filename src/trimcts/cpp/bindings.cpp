
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>     // For map/vector conversions
#include <pybind11/pytypes.h> // For py::object, py::handle

#include "mcts.h"             // Include your MCTS logic header
#include "config.h"           // Include your config struct header
#include "python_interface.h" // For types

namespace py = pybind11;
namespace tc = trimcts; // Alias for your C++ namespace

// Helper function to transfer config from Python Pydantic model to C++ struct
tc::SearchConfig python_to_cpp_config(const py::object &py_config)
{
  tc::SearchConfig cpp_config;
  // Use py::getattr with default values or checks
  if (py::hasattr(py_config, "max_simulations"))
  {
    cpp_config.max_simulations = py_config.attr("max_simulations").cast<uint32_t>();
  }
  if (py::hasattr(py_config, "max_depth"))
  {
    cpp_config.max_depth = py_config.attr("max_depth").cast<uint32_t>();
  }
  if (py::hasattr(py_config, "cpuct"))
  {
    cpp_config.cpuct = py_config.attr("cpuct").cast<double>();
  }
  if (py::hasattr(py_config, "dirichlet_alpha"))
  {
    cpp_config.dirichlet_alpha = py_config.attr("dirichlet_alpha").cast<double>();
  }
  if (py::hasattr(py_config, "dirichlet_epsilon"))
  {
    cpp_config.dirichlet_epsilon = py_config.attr("dirichlet_epsilon").cast<double>();
  }
  if (py::hasattr(py_config, "discount"))
  {
    cpp_config.discount = py_config.attr("discount").cast<double>();
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

  // Optional: Expose the C++ Node class if needed for debugging or advanced usage
  /*
  py::class_<tc::Node>(m, "NodeCpp")
      .def(py::init<py::object, tc::Node*, tc::Action, float>(),
           py::arg("state"), py::arg("parent") = nullptr, py::arg("action") = -1, py::arg("prior") = 0.0f)
      .def_property_readonly("visit_count", [](const tc::Node &n){ return n.visit_count_; })
      .def_property_readonly("value_estimate", &tc::Node::get_value_estimate)
      // Add other methods/properties as needed
      ;
  */

  // Optional: Expose the C++ SearchConfig struct if needed
  /*
   py::class_<tc::SearchConfig>(m, "SearchConfigCpp")
      .def(py::init<>())
      .def_readwrite("max_simulations", &tc::SearchConfig::max_simulations)
      .def_readwrite("max_depth", &tc::SearchConfig::max_depth)
      .def_readwrite("cpuct", &tc::SearchConfig::cpuct)
      // ... other fields
      ;
  */

#ifdef VERSION_INFO
  m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
  m.attr("__version__") = "dev";
#endif
}
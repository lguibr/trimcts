// File: src/trimcts/cpp/bindings.cpp

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>     // For map/vector conversions
#include <pybind11/pytypes.h> // For py::object, py::handle

#include "mcts.h"             // C++ MCTS logic header
#include "config.h"           // C++ SearchConfig struct
#include "python_interface.h" // For NetworkOutput, VisitMap, etc.
#include <string>             // std::string
#include <stdexcept>          // std::runtime_error

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
    // **NEW**: read the Python-configured batch size
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

// Wrapper function exposed to Python
tc::VisitMap run_mcts_cpp_wrapper(
    py::object root_state_py,
    py::object network_interface_py,
    const py::object &config_py // Pass Python config object
)
{
  // Convert Python config to C++ config struct (including batch size)
  tc::SearchConfig config_cpp = python_to_cpp_config(config_py);

  // Call the internal C++ MCTS implementation
  try
  {
    return tc::run_mcts_cpp_internal(root_state_py, network_interface_py, config_cpp);
  }
  catch (const std::exception &e)
  {
    // Convert C++ exceptions to Python exceptions
    throw py::value_error(std::string("Error in C++ MCTS execution: ") + e.what());
  }
  catch (const py::error_already_set &)
  {
    // Propagate any Python-side exceptions
    throw;
  }
}

PYBIND11_MODULE(trimcts_cpp, m)
{
  m.doc() = "C++ core module for TriMCTS";

  m.def("run_mcts_cpp",
        &run_mcts_cpp_wrapper,
        py::arg("root_state"),
        py::arg("network_interface"),
        py::arg("config"),
        "Runs MCTS simulations from the root state using the provided network interface and configuration (C++).");

#ifdef VERSION_INFO
  m.attr("__version__") = VERSION_INFO;
#else
  m.attr("__version__") = "dev";
#endif
}

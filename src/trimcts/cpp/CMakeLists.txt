# File: src/trimcts/cpp/CMakeLists.txt

cmake_minimum_required(VERSION 3.14)
project(trimcts_cpp LANGUAGES CXX)

# Locate Pybind11 using variables passed from setup.py (pybind11_DIR)
# Pybind11's config should handle finding the correct Python components
find_package(pybind11 CONFIG REQUIRED)

# REMOVED: find_package(Python 3.10 COMPONENTS Interpreter Development REQUIRED)
# We will rely on pybind11_add_module to handle Python discovery and linking.

# Sources
set(TRIMCTS_SOURCES
    bindings.cpp
    mcts.cpp
    mcts_manager.cpp # ADDED missing source file
)

# Build the pybind11 module
# This command uses pybind11's logic to find and link Python
pybind11_add_module(trimcts_cpp MODULE ${TRIMCTS_SOURCES})

# C++17 Standard
target_compile_features(trimcts_cpp PRIVATE cxx_std_17)

# Optimisation flags and visibility
if(MSVC)
  target_compile_options(trimcts_cpp PRIVATE /O2)
else()
  target_compile_options(trimcts_cpp PRIVATE -O3 -DNDEBUG)
  # Symbol visibility for non-Apple Unix-like systems
  if(NOT APPLE)
    target_compile_options(trimcts_cpp PRIVATE -fvisibility=hidden)
  endif()
endif()

# Output directory is now set via CMAKE_LIBRARY_OUTPUT_DIRECTORY in setup.py

# --- Status Messages ---
# Informational messages about what pybind11 found might be useful if needed later,
# but let's keep it clean for now.
message(STATUS "pybind11 Include Dirs: ${pybind11_INCLUDE_DIRS}") # From find_package(pybind11)
message(STATUS "Building C++ extension for TriMCTS version ${TRIMCTS_VERSION_INFO}")
# File: src/trimcts/cpp/CMakeLists.txt

cmake_minimum_required(VERSION 3.14)
project(trimcts_cpp LANGUAGES CXX)

# Locate Pybind11 using variables passed from setup.py (pybind11_DIR)
# Pybind11's config should handle finding the correct Python components
find_package(pybind11 CONFIG REQUIRED)

# Find Python - simplified request, primarily for informational purposes
# and potentially setting CMAKE_CXX_STANDARD if not set elsewhere.
# Pybind11 targets should handle the actual Python dependency linking.
find_package(Python 3.10 COMPONENTS Interpreter Development REQUIRED)

# Sources
set(TRIMCTS_SOURCES
    bindings.cpp
    mcts.cpp
)

# Build the pybind11 module
# This command uses pybind11's logic to find and link Python
pybind11_add_module(trimcts_cpp MODULE ${TRIMCTS_SOURCES})

# Link against Python libraries (handled by pybind11_add_module)
# target_link_libraries(trimcts_cpp PRIVATE Python::Module) # Usually not needed when using pybind11_add_module

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
# Use Python_EXECUTABLE variable set by FindPython
if(Python_EXECUTABLE)
  message(STATUS "Python Executable (found by CMake): ${Python_EXECUTABLE}")
else()
  message(WARNING "Python Executable not found by CMake's FindPython.")
endif()
# Use Python_INCLUDE_DIRS variable set by FindPython
if(Python_INCLUDE_DIRS)
  message(STATUS "Python Include Dirs (found by CMake): ${Python_INCLUDE_DIRS}")
else()
  message(WARNING "Python Include Dirs not found by CMake's FindPython.")
endif()
# Use Python_LIBRARIES variable set by FindPython
if(Python_LIBRARIES)
  message(STATUS "Python Libraries (found by CMake): ${Python_LIBRARIES}")
else()
  message(WARNING "Python Libraries not found by CMake's FindPython.")
endif()

message(STATUS "pybind11 Include Dirs: ${pybind11_INCLUDE_DIRS}") # From find_package(pybind11)
message(STATUS "Building C++ extension for TriMCTS version ${TRIMCTS_VERSION_INFO}")
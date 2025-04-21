
# File: src/trimcts/cpp/CMakeLists.txt

cmake_minimum_required(VERSION 3.14)
project(trimcts_cpp LANGUAGES CXX)

# Locate Python and Pybind11 using variables passed from setup.py
# These hints help CMake find the correct Python installation from the venv
find_package(Python REQUIRED COMPONENTS Interpreter Development)
find_package(pybind11 CONFIG REQUIRED)

# Sources
set(TRIMCTS_SOURCES
    bindings.cpp
    mcts.cpp
)

# Build the pybind11 module
pybind11_add_module(trimcts_cpp MODULE ${TRIMCTS_SOURCES})

# Link against Python libraries
target_link_libraries(trimcts_cpp PRIVATE Python::Module)

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
# No need to set target properties here for output location.
message(STATUS "Python Executable: ${Python_EXECUTABLE}")
message(STATUS "Python Include Dir: ${Python_INCLUDE_DIR}")
message(STATUS "Python Libraries: ${Python_LIBRARIES}")
message(STATUS "pybind11 Include Dirs: ${pybind11_INCLUDE_DIRS}")
message(STATUS "Building C++ extension for TriMCTS version ${TRIMCTS_VERSION_INFO}")
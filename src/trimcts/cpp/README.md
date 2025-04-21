
# `src/trimcts/cpp` - C++ Core Implementation

This directory houses the C++ source code for the high-performance Monte Carlo Tree Search (MCTS) engine used by the `trimcts` package.

## Contents

-   [`CMakeLists.txt`](CMakeLists.txt): The CMake build script responsible for configuring the build process, finding dependencies (Python, Pybind11), and defining the C++ extension module target (`trimcts_cpp`).
-   [`bindings.cpp`](bindings.cpp): Contains the Pybind11 code that creates the Python bindings for the C++ functions and classes, exposing them to the Python `trimcts` package. It handles the conversion between Python objects (like the game state, network interface, and configuration) and their C++ counterparts.
-   [`config.h`](config.h): Defines the C++ `SearchConfig` struct, which mirrors the Python `SearchConfiguration` Pydantic model, holding MCTS parameters used by the C++ core.
-   [`mcts.h`](mcts.h): The header file for the MCTS implementation. It declares the `Node` class representing a node in the search tree and the signature for the core MCTS function (`run_mcts_cpp_internal`).
-   [`mcts.cpp`](mcts.cpp): The main implementation file for the MCTS algorithm. It contains the logic for the `Node` class methods (selection, expansion, backpropagation, PUCT calculation, Dirichlet noise) and the `run_mcts_cpp_internal` function orchestrating the search process.
-   [`python_interface.h`](python_interface.h): Provides helper functions to facilitate interaction between C++ and Python objects. It includes functions to call methods on the Python game state object (like `copy`, `step`, `is_over`, `valid_actions`) and the Python network interface object (`evaluate_state`, `evaluate_batch`).

## Overview

The C++ core is designed for performance critical parts of the MCTS algorithm:

-   **Tree Traversal:** Efficiently navigating the search tree using the PUCT formula.
-   **Node Management:** Creating, storing, and updating nodes within the tree.
-   **Simulation Loop:** Executing the core select-expand-evaluate-backpropagate loop for the specified number of simulations.

It interacts with Python for:

-   **Game Logic:** Calling methods on the Python `GameState` object provided by the user (via `trianglengin` or a compatible implementation).
-   **Neural Network Evaluation:** Calling methods on the Python network interface object provided by the user to get policy and value predictions.

The `bindings.cpp` file acts as the bridge, managed by Pybind11, allowing seamless calls between the Python wrapper code in [`src/trimcts`](../README.md) and this C++ core.

Refer to the main project [README.md](../../../README.md) for build instructions.

This directory contains the core Python source code for the `trimcts` package.

## Contents

-   [`__init__.py`](__init__.py): Exposes the public API of the package, including `run_mcts`, `SearchConfiguration`, and the network interfaces.
-   [`config.py`](config.py): Defines the Pydantic model `SearchConfiguration` for MCTS parameters.
-   [`mcts_wrapper.py`](mcts_wrapper.py): Contains the Python entry point `run_mcts`, the network interface protocols (`AlphaZeroNetworkInterface`, `MuZeroNetworkInterface`), and handles the interaction with the C++ extension module.
-   [`py.typed`](py.typed): A marker file indicating that this package provides type information (PEP 561 compliant), allowing type checkers like MyPy to verify its usage.
-   [`cpp/`](cpp/README.md): Contains the C++ source code and build configuration for the high-performance MCTS core. ([Link to C++ README](cpp/README.md))

## Overview

The Python code here primarily serves as:

1.  **Configuration Layer:** Providing a user-friendly way to define MCTS settings via `SearchConfiguration`.
2.  **Interface Layer:** Defining the expected interfaces (`AlphaZeroNetworkInterface`, `MuZeroNetworkInterface`) that user-provided neural network wrappers must adhere to.
3.  **Binding Layer:** Calling the compiled C++ extension (`trimcts_cpp`) via the `run_mcts` function, passing the game state, network wrapper, and configuration.
4.  **Type Safety:** Providing type hints for static analysis and improved developer experience.

See the main project [README.md](../../README.md) for installation and usage examples.
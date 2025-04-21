
# `tests` - Python Unit and Integration Tests

This directory contains the automated tests for the `trimcts` package, primarily using the `pytest` framework.

## Contents

-   [`conftest.py`](conftest.py): (If present) Contains shared fixtures, hooks, or plugins used across multiple test files within this directory.
-   [`test_alpha_wrapper.py`](test_alpha_wrapper.py): Contains tests specifically focused on the AlphaZero MCTS functionality. This includes:
    -   Testing the `run_mcts` function with a mock `AlphaZeroNetworkInterface`.
    -   Verifying behavior with and without Dirichlet noise.
    -   Testing edge cases like running MCTS on a terminal state.
    -   Integration tests using `trianglengin`'s `GameState` if available, otherwise using mock objects.

## Running Tests

Tests are crucial for ensuring the correctness and stability of the package, especially given the interaction between Python and C++.

To run the tests:

1.  Make sure you have installed the package in editable mode with development dependencies:
    ```bash
    pip install -e .[dev]
    ```
2.  Navigate to the root directory of the project (the one containing `pyproject.toml`).
3.  Run `pytest`:
    ```bash
    pytest
    ```

This command will automatically discover and execute the tests defined in this directory. Test coverage reports are configured in [`pyproject.toml`](../pyproject.toml).

Refer to the main project [README.md](../README.md) for more details on the project structure and building.
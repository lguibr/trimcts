
[![CI](https://github.com/lguibr/trimcts/actions/workflows/ci_cd.yml/badge.svg)](https://github.com/lguibr/trimcts/actions)
[![PyPI](https://img.shields.io/pypi/v/trimcts.svg)](https://pypi.org/project/trimcts/)
[![Coverage Status](https://codecov.io/gh/lguibr/trimcts/graph/badge.svg?token=YOUR_CODECOV_TOKEN_HERE)](https://codecov.io/gh/lguibr/trimcts) <!-- TODO: Add Codecov token -->
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

# TriMCTS

<img src="bitmap.png" alt="TriMCTS Logo" width="300"/>


**TriMCTS** is an installable Python package providing C++ bindings for Monte Carlo Tree Search, supporting both AlphaZero and MuZero paradigms, optimized for triangular grid games like the one in `trianglengin`.

## 🔑 Key Features

-   High-performance C++ core implementation.
-   Seamless Python integration via Pybind11.
-   Supports AlphaZero-style evaluation (policy/value from state).
-   **Batched Network Evaluations:** Efficiently calls the Python network's `evaluate_batch` method during search for improved performance, especially with GPUs.
-   **MCTS Tree Reuse:** Significantly speeds up sequential MCTS calls (e.g., during self-play) by reusing the relevant subtree from the previous search step. The C++ core manages the tree lifetime via opaque handles (`py::capsule`) passed between Python calls.
-   (Planned) Supports MuZero-style evaluation (initial inference + recurrent inference).
-   Configurable search parameters (simulation count, PUCT, discount factor, Dirichlet noise, batch size).
-   Designed for use with external Python game state objects and network evaluators.
-   Type-hinted Python API (`py.typed` compliant).

## 🚀 Installation

```bash
# From PyPI (once published)
pip install trimcts

# For development (from cloned repo root)
# Ensure you clean previous builds if you encounter issues:
# rm -rf build/ src/trimcts.egg-info/ dist/ src/trimcts/trimcts_cpp.*.so
pip install -e .[dev]
```

## 💡 Usage Example (AlphaZero Style with Tree Reuse)

```python
import time
import numpy as np
import torch # Added import
# Use the actual GameState if trianglengin is installed
try:
    from trianglengin import GameState, EnvConfig
    HAS_TRIANGLENGIN = True
except ImportError:
    # Define minimal mocks if trianglengin is not available
    class GameState: # type: ignore
        def __init__(self, *args, **kwargs): self.current_step = 0
        def is_over(self): return False
        def copy(self): return self
        def step(self, action): return 0.0, False
        def get_outcome(self): return 0.0
        def valid_actions(self): return [0, 1]
    class EnvConfig: pass # type: ignore
    HAS_TRIANGLENGIN = False

# Assuming alphatriangle is installed and provides these:
# from alphatriangle.nn import NeuralNetwork # Example network wrapper
# from alphatriangle.config import ModelConfig, TrainConfig

from trimcts import run_mcts, SearchConfiguration, AlphaZeroNetworkInterface

# --- Mock Neural Network (same as before) ---
class MockNeuralNetwork:
    def __init__(self, *args, **kwargs):
        self.model = torch.nn.Module() # Dummy model
        print("MockNeuralNetwork initialized.")

    def evaluate_state(self, state: GameState) -> tuple[dict[int, float], float]:
        valid_actions = state.valid_actions()
        if not valid_actions: return {}, 0.0
        policy = {action: 1.0 / len(valid_actions) for action in valid_actions}
        value = 0.5
        return policy, value

    def evaluate_batch(self, states: list[GameState]) -> list[tuple[dict[int, float], float]]:
        print(f"  Mock evaluate_batch called with {len(states)} states.")
        return [self.evaluate_state(s) for s in states]

    def load_weights(self, path): print(f"Mock: Pretending to load weights from {path}")
    def to(self, device): print(f"Mock: Pretending to move model to {device}"); return self
# --- End Mock Neural Network ---

# --- AlphaZero Wrapper (same as before) ---
class MyAlphaZeroWrapper(AlphaZeroNetworkInterface):
    def __init__(self, model_path: str | None = None):
        self.network = MockNeuralNetwork()
        if model_path: self.network.load_weights(model_path)
        self.network.model.eval()
        print("MyAlphaZeroWrapper initialized.")

    def evaluate_state(self, state: GameState) -> tuple[dict[int, float], float]:
        print(f"Python: Evaluating SINGLE state step {state.current_step}")
        policy_map, value = self.network.evaluate_state(state)
        print(f"Python: Single evaluation result - Policy keys: {len(policy_map)}, Value: {value:.4f}")
        return policy_map, value

    def evaluate_batch(self, states: list[GameState]) -> list[tuple[dict[int, float], float]]:
        print(f"Python: Evaluating BATCH of {len(states)} states.")
        results = self.network.evaluate_batch(states)
        print(f"Python: Batch evaluation returned {len(results)} results.")
        return results

# --- Simulation Loop Example ---
env_config = EnvConfig()
if HAS_TRIANGLENGIN:
    env_config.ROWS = 3
    env_config.COLS = 3
    env_config.NUM_SHAPE_SLOTS = 1
    env_config.PLAYABLE_RANGE_PER_ROW = [(0,3), (0,3), (0,3)]

game_state = GameState(config=env_config, initial_seed=42)
network_wrapper = MyAlphaZeroWrapper()

mcts_config = SearchConfiguration()
mcts_config.max_simulations = 50
mcts_config.mcts_batch_size = 8

# --- Tree Reuse Variables ---
mcts_tree_handle = None # Start with no tree
last_action = -1        # No previous action initially

print("--- Running Self-Play Loop with Tree Reuse ---")
max_episode_steps = 10
for step in range(max_episode_steps):
    if game_state.is_over():
        print(f"\nGame over at step {step}. Final Score: {game_state.game_score()}")
        break

    print(f"\n--- Step {step} ---")
    print(f"Current State Step: {game_state.current_step}")
    print(f"Passing tree handle: {'Yes' if mcts_tree_handle else 'No'}")
    print(f"Passing last action: {last_action}")

    # Run MCTS, passing the handle and last action
    # It returns visit counts AND the new handle
    start_time = time.time()
    visit_counts, mcts_tree_handle = run_mcts(
        root_state=game_state,
        network_interface=network_wrapper,
        config=mcts_config,
        previous_tree_handle=mcts_tree_handle, # Pass handle from previous step
        last_action=last_action               # Pass action that led to current state
    )
    end_time = time.time()
    print(f"MCTS Result (Visit Counts) after {end_time - start_time:.3f} seconds:")
    print(visit_counts)
    print(f"Received new tree handle: {'Present' if mcts_tree_handle else 'None'}")

    # Select best action based on visits
    if not visit_counts:
        print("MCTS returned no visits. Ending episode.")
        break
    best_action = max(visit_counts, key=visit_counts.get)
    print(f"Selected Action: {best_action}")

    # Store the selected action for the *next* MCTS call
    last_action = best_action

    # Apply the action to the game state
    reward, done = game_state.step(best_action)
    print(f"Step Reward: {reward:.3f}, Done: {done}")

else:
     print(f"\nEpisode finished after {max_episode_steps} steps.")

# The mcts_tree_handle (a py::capsule) will be automatically garbage collected
# by Python when it goes out of scope, triggering the C++ destructor.
print("\n--- End of Simulation ---")

```

*(MuZero example will be added later)*

## 📂 Project Structure

```
trimcts/
├── .github/workflows/      # CI configuration (e.g., ci_cd.yml)
├── src/trimcts/            # Python package source ([src/trimcts/README.md](src/trimcts/README.md))
│   ├── cpp/                # C++ source code ([src/trimcts/cpp/README.md](src/trimcts/cpp/README.md))
│   │   ├── CMakeLists.txt  # CMake build script for C++ part
│   │   ├── bindings.cpp    # Pybind11 bindings
│   │   ├── config.h        # C++ configuration struct
│   │   ├── mcts.cpp        # C++ MCTS implementation (Node, simulation loop)
│   │   ├── mcts.h          # C++ MCTS header
│   │   ├── mcts_manager.cpp # C++ MCTS Tree Manager implementation
│   │   ├── mcts_manager.h  # C++ MCTS Tree Manager header (handles lifetime)
│   │   ├── python_interface.h # C++ helpers for Python interaction
│   │   └── structs.h       # Common C++ structs (NetworkOutput, etc.)
│   ├── __init__.py         # Exposes public API (run_mcts, configs, etc.)
│   ├── config.py           # Python SearchConfiguration (Pydantic)
│   ├── mcts_wrapper.py     # Python network interface definition & run_mcts wrapper
│   └── py.typed            # Marker file for type checkers (PEP 561)
├── tests/                  # Python tests ([tests/README.md](tests/README.md))
│   ├── conftest.py
│   └── test_alpha_wrapper.py # Tests for AlphaZero functionality
├── .gitignore
├── LICENSE
├── MANIFEST.in             # Specifies files for source distribution
├── pyproject.toml          # Build system & package configuration
├── README.md               # This file
└── setup.py                # Setup script for C++ extension building
```

## 🛠️ Building from Source

1.  Clone the repository: `git clone https://github.com/lguibr/trimcts.git`
2.  Navigate to the directory: `cd trimcts`
3.  **Recommended:** Create and activate a virtual environment:
    ```bash
    python -m venv .venv
    source .venv/bin/activate # On Windows use `.venv\Scripts\activate`
    ```
4.  Install build dependencies: `pip install pybind11>=2.10 cmake wheel`
5.  **Clean previous builds (important if switching Python versions or encountering issues):**
    ```bash
    rm -rf build/ src/trimcts.egg-info/ dist/ src/trimcts/trimcts_cpp.*.so
    ```
6.  Install the package in editable mode: `pip install -e .`

## 🧪 Running Tests

```bash
# Make sure you have installed dev dependencies
pip install -e .[dev]
pytest
```

## 🤝 Contributing

Contributions are welcome! Please follow standard fork-and-pull-request workflow. Ensure tests pass and code adheres to formatting/linting standards (Ruff, MyPy).

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
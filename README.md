
[![CI](https://github.com/lguibr/trimcts/actions/workflows/ci.yml/badge.svg)](https://github.com/lguibr/trimcts/actions)
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
-   (Planned) Supports MuZero-style evaluation (initial inference + recurrent inference).
-   Configurable search parameters (simulation count, PUCT, discount factor, Dirichlet noise).
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

## 💡 Usage Example (AlphaZero Style)

```python
import time
import numpy as np
import torch # Added import
from trianglengin.core.environment import GameState # Example state object
from trianglengin.config import EnvConfig
# Assuming alphatriangle is installed and provides these:
# from alphatriangle.nn import NeuralNetwork # Example network wrapper
# from alphatriangle.config import ModelConfig, TrainConfig

from trimcts import run_mcts, SearchConfiguration, AlphaZeroNetworkInterface

# --- Mock Neural Network for demonstration ---
# Replace with your actual network implementation
class MockNeuralNetwork:
    def __init__(self, *args, **kwargs):
        self.model = torch.nn.Module() # Dummy model
        print("MockNeuralNetwork initialized.")

    def evaluate(self, state: GameState) -> tuple[dict[int, float], float]:
        # Mock evaluation: uniform policy over valid actions, fixed value
        valid_actions = state.valid_actions()
        if not valid_actions:
            return {}, 0.0 # Terminal or no valid actions
        policy = {action: 1.0 / len(valid_actions) for action in valid_actions}
        value = 0.5 # Fixed mock value
        return policy, value

    def evaluate_batch(self, states: list[GameState]) -> list[tuple[dict[int, float], float]]:
        return [self.evaluate(s) for s in states]

    def load_weights(self, path):
        print(f"Mock: Pretending to load weights from {path}")

    def to(self, device):
        print(f"Mock: Pretending to move model to {device}")
        return self
# --- End Mock Neural Network ---


# 1. Define your AlphaZero network wrapper conforming to the interface
class MyAlphaZeroWrapper(AlphaZeroNetworkInterface):
    def __init__(self, model_path: str | None = None):
        # Load your PyTorch/TensorFlow/etc. model here
        # Example using a Mock NeuralNetwork
        env_cfg = EnvConfig()
        # model_cfg = ModelConfig() # Assuming these exist if using alphatriangle
        # train_cfg = TrainConfig(DEVICE="cpu")
        self.network = MockNeuralNetwork() # Using Mock for this example
        # Load weights if model_path is provided
        if model_path:
             self.network.load_weights(model_path)
        # self.network.to(torch.device("cpu")) # Ensure model is on correct device if using real NN
        self.network.model.eval() # Set to evaluation mode
        print("MyAlphaZeroWrapper initialized.")

    def evaluate_state(self, state: GameState) -> tuple[dict[int, float], float]:
        """
        Evaluates a single game state.

        Args:
            state: The GameState object (passed from C++).

        Returns:
            A tuple containing:
                - Policy dict: {action_index: probability}
                - Value estimate: float
        """
        print(f"Python: Evaluating state step {state.current_step}")
        # Use the evaluate method of your network wrapper
        # Add necessary state transformations if your network expects specific input format
        # e.g., state_tensor = self.transform_state(state)
        # policy_logits, value_logit = self.network.model(state_tensor)
        # policy_map = self.process_policy(policy_logits, state.valid_actions())
        # value = torch.tanh(value_logit).item() # Example processing
        policy_map, value = self.network.evaluate(state) # Using mock evaluate directly
        print(f"Python: Evaluation result - Policy keys: {len(policy_map)}, Value: {value:.4f}")
        return policy_map, value

    def evaluate_batch(self, states: list[GameState]) -> list[tuple[dict[int, float], float]]:
        """
        Evaluates a batch of game states.

        Args:
            states: A list of GameState objects.

        Returns:
            A list of tuples, each containing (policy_dict, value_estimate).
        """
        print(f"Python: Evaluating batch of {len(states)} states.")
        # Use the evaluate_batch method of your network wrapper
        # Add necessary state transformations and batching if needed
        # e.g., batch_tensor = self.transform_batch(states)
        # policy_logits_batch, value_logit_batch = self.network.model(batch_tensor)
        # results = self.process_batch_output(policy_logits_batch, value_logit_batch, states)
        results = self.network.evaluate_batch(states) # Using mock evaluate_batch directly
        print(f"Python: Batch evaluation returned {len(results)} results.")
        return results

# 2. Instantiate your game state and network wrapper
env_config = EnvConfig()
# Ensure the config creates a playable state for the example
env_config.ROWS = 3
env_config.COLS = 3
env_config.NUM_SHAPE_SLOTS = 1
env_config.PLAYABLE_RANGE_PER_ROW = [(0,3), (0,3), (0,3)] # Example playable range

root_state = GameState(config=env_config, initial_seed=42)
network_wrapper = MyAlphaZeroWrapper() # Add path to your trained model if needed

# 3. Configure MCTS parameters
mcts_config = SearchConfiguration()
mcts_config.max_simulations = 50
mcts_config.max_depth = 10
mcts_config.cpuct = 1.25
mcts_config.dirichlet_alpha = 0.3
mcts_config.dirichlet_epsilon = 0.25
mcts_config.discount = 1.0 # AlphaZero typically uses no discount during search

# 4. Run MCTS
# The C++ run_mcts function will call network_wrapper.evaluate_batch() or evaluate_state()
print("Running MCTS...")
# Ensure root_state is not terminal before running
if not root_state.is_over():
    # run_mcts returns a dictionary: {action: visit_count}
    start_time = time.time()
    visit_counts = run_mcts(root_state, network_wrapper, mcts_config)
    end_time = time.time()
    print(f"\nMCTS Result (Visit Counts) after {end_time - start_time:.2f} seconds:")
    print(visit_counts)

    # Example: Select best action based on visits
    if visit_counts:
        best_action = max(visit_counts, key=visit_counts.get)
        print(f"\nBest action based on visits: {best_action}")
    else:
        print("\nNo actions explored or MCTS failed.")
else:
    print("Root state is already terminal. Cannot run MCTS.")

```

*(MuZero example will be added later)*

## 📂 Project Structure

```
trimcts/
├── .github/workflows/      # CI configuration (e.g., ci.yml)
├── src/trimcts/            # Python package source ([src/trimcts/README.md](src/trimcts/README.md))
│   ├── cpp/                # C++ source code ([src/trimcts/cpp/README.md](src/trimcts/cpp/README.md))
│   │   ├── CMakeLists.txt  # CMake build script for C++ part
│   │   ├── bindings.cpp    # Pybind11 bindings
│   │   ├── config.h        # C++ configuration struct
│   │   ├── mcts.cpp        # C++ MCTS implementation
│   │   ├── mcts.h          # C++ MCTS header
│   │   └── python_interface.h # C++ helpers for Python interaction
│   ├── __init__.py         # Exposes public API (run_mcts, configs, etc.)
│   ├── config.py           # Python SearchConfiguration (Pydantic)
│   ├── mcts_wrapper.py     # Python network interface definition
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
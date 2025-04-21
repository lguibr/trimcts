# File: tests/test_alpha_wrapper.py
import time
from typing import Any, TypeAlias  # Import TypeAlias

import pytest

# --- Define GameState and EnvConfig types ---
# Try importing the real ones first
try:
    from trianglengin.config import EnvConfig as RealEnvConfig
    from trianglengin.core.environment import GameState as RealGameState

    HAS_TRIANGLENGIN = True
    # Define type aliases pointing to the real classes
    GameState: TypeAlias = RealGameState
    EnvConfig: TypeAlias = RealEnvConfig
except ImportError:
    HAS_TRIANGLENGIN = False

    # Define minimal mock classes if trianglengin is not available
    class MockGameStateForTest:
        def __init__(self, *_args: Any, **_kwargs: Any):  # Prefix unused args
            self.step_count = 0
            self._is_over = False  # Add state for forcing terminal
            self.current_step = 0  # Add current_step for logging in wrapper

        def copy(self) -> "MockGameStateForTest":
            s = MockGameStateForTest()
            s.step_count = self.step_count
            s._is_over = self._is_over
            s.current_step = self.current_step
            return s

        # Prefix unused action
        def step(self, _action: int) -> tuple[float, bool]:
            if self._is_over:
                return 0.0, True  # No change if already over
            self.step_count += 1
            self.current_step += 1
            self._is_over = self.step_count >= 5  # Example terminal condition
            return 0.0, self._is_over

        def is_over(self) -> bool:
            return self._is_over or self.step_count >= 5

        def get_outcome(self) -> float:
            return -1.0 if self.is_over() else 0.0  # Use -1 for loss

        def valid_actions(self) -> list[int]:
            return [] if self.is_over() else [0, 1]

        def force_game_over(self, reason: str) -> None:  # Add method for testing
            print(f"Mock force_game_over: {reason}")
            self._is_over = True

    class MockEnvConfigForTest:
        pass

    # Define type aliases pointing to the mock classes
    GameState: TypeAlias = MockGameStateForTest  # type: ignore # Allow redefinition for mock case
    EnvConfig: TypeAlias = MockEnvConfigForTest  # type: ignore # Allow redefinition for mock case
# --- End GameState/EnvConfig Definition ---


# Now import trimcts components AFTER GameState/EnvConfig are defined
from trimcts import AlphaZeroNetworkInterface, SearchConfiguration, run_mcts


# --- Dummy Network Implementation ---
# This class now uses the GameState type alias defined above
class DummyAlphaNetwork(AlphaZeroNetworkInterface):
    """A simple network that returns fixed policy/value."""

    def __init__(self, action_dim: int = 2, value: float = 0.5, delay: float = 0.0):
        self.action_dim = action_dim
        self.value = value
        self.policy = dict.fromkeys(range(action_dim), 1.0 / action_dim)
        self.eval_count = 0
        self.batch_eval_count = 0
        self.delay = delay  # Simulate network inference time

    def evaluate_state(self, state: GameState) -> tuple[dict[int, float], float]:
        self.eval_count += 1
        if self.delay > 0:
            time.sleep(self.delay)
        # Return policy only for valid actions in the current state
        valid_actions = state.valid_actions()
        valid_policy = {a: p for a, p in self.policy.items() if a in valid_actions}
        policy_sum = sum(valid_policy.values())
        if policy_sum > 1e-6:
            normalized_policy = {a: p / policy_sum for a, p in valid_policy.items()}
        else:  # Handle case where no valid actions have policy prob
            normalized_policy = {}
            if (
                valid_actions
            ):  # Assign uniform if valid actions exist but policy was zero
                uniform_prob = 1.0 / len(valid_actions)
                normalized_policy = dict.fromkeys(valid_actions, uniform_prob)

        return normalized_policy, self.value

    def evaluate_batch(
        self, states: list[GameState]
    ) -> list[tuple[dict[int, float], float]]:
        self.batch_eval_count += 1
        if self.delay > 0:
            time.sleep(self.delay * 0.1 * len(states))  # Simulate some batch overhead
        return [self.evaluate_state(s) for s in states]


# --- Test Fixtures ---
@pytest.fixture
def dummy_state() -> GameState:
    """Provides a simple dummy game state."""
    if HAS_TRIANGLENGIN:
        # Use the real classes if available
        # Define a playable range consistent with ROWS=3, COLS=3
        test_playable_range = [(0, 3), (0, 3), (0, 3)]
        test_config = RealEnvConfig(
            ROWS=3,
            COLS=3,
            NUM_SHAPE_SLOTS=1,
            PLAYABLE_RANGE_PER_ROW=test_playable_range,  # Provide the range
        )
        return RealGameState(config=test_config, initial_seed=1)
    else:
        # Use the mock classes (via the GameState alias)
        return GameState()


@pytest.fixture
def dummy_network() -> DummyAlphaNetwork:
    """Provides a dummy network interface."""
    # Determine action dim based on whether real or mock is used
    # For now, let's stick to action_dim=2 for the dummy network as defined.
    return DummyAlphaNetwork(action_dim=2, value=0.1)


@pytest.fixture
def search_config() -> SearchConfiguration:
    """Provides a default search configuration."""
    return SearchConfiguration(
        max_simulations=16,  # Keep low for testing
        max_depth=5,
        cpuct=1.25,
        dirichlet_alpha=0.3,
        dirichlet_epsilon=0.25,
        discount=1.0,
    )


# --- Tests ---


def test_mcts_run_alpha_basic(
    dummy_state: GameState,
    dummy_network: DummyAlphaNetwork,
    search_config: SearchConfiguration,
) -> None:
    """Test basic MCTS run with AlphaZero interface."""
    search_config.max_simulations = 10
    search_config.dirichlet_alpha = (
        0.0  # Disable noise for predictable results initially
    )

    if dummy_state.is_over():
        pytest.skip("Initial dummy state is already terminal.")

    print("\n--- Starting Basic MCTS Run ---")
    start_time = time.time()
    visit_counts = run_mcts(dummy_state, dummy_network, search_config)
    duration = time.time() - start_time
    print(f"--- MCTS Run Finished ({duration:.4f}s) ---")
    print(f"Visit Counts: {visit_counts}")
    print(f"Network single evals: {dummy_network.eval_count}")
    print(f"Network batch evals: {dummy_network.batch_eval_count}")

    assert isinstance(visit_counts, dict)
    # Check if actions explored are valid for the root state
    valid_root_actions = set(dummy_state.valid_actions())
    assert set(visit_counts.keys()).issubset(valid_root_actions)
    assert all(isinstance(v, int) and v >= 0 for v in visit_counts.values())
    # With 10 sims, we expect some visits
    assert sum(visit_counts.values()) > 0
    # Since policy is uniform and value is constant, visits might be somewhat balanced
    # This is a weak check
    if len(valid_root_actions) > 0:
        assert len(visit_counts) > 0


def test_mcts_run_alpha_with_noise(
    dummy_state: GameState,
    dummy_network: DummyAlphaNetwork,
    search_config: SearchConfiguration,
) -> None:
    """Test MCTS run with Dirichlet noise enabled."""
    search_config.max_simulations = 32  # More sims to see noise effect
    search_config.dirichlet_alpha = 0.5
    search_config.dirichlet_epsilon = 0.25

    if dummy_state.is_over():
        pytest.skip("Initial dummy state is already terminal.")

    print("\n--- Starting MCTS Run with Noise ---")
    start_time = time.time()
    visit_counts = run_mcts(dummy_state, dummy_network, search_config)
    duration = time.time() - start_time
    print(f"--- MCTS Run Finished ({duration:.4f}s) ---")
    print(f"Visit Counts: {visit_counts}")

    assert isinstance(visit_counts, dict)
    valid_root_actions = set(dummy_state.valid_actions())
    assert set(visit_counts.keys()).issubset(valid_root_actions)
    assert all(isinstance(v, int) and v >= 0 for v in visit_counts.values())
    assert sum(visit_counts.values()) > 0


def test_mcts_run_on_terminal_state(
    dummy_state: GameState,
    dummy_network: DummyAlphaNetwork,
    search_config: SearchConfiguration,
) -> None:
    """Test MCTS run starting from a terminal state (should return empty)."""
    # Make the state terminal
    if HAS_TRIANGLENGIN:
        # Use the real GameState's method to force game over
        dummy_state.force_game_over("Forced terminal for test")
    else:
        # Use the mock state's method
        if hasattr(dummy_state, "force_game_over") and callable(
            getattr(dummy_state, "force_game_over")
        ):
            dummy_state.force_game_over("Forced terminal for mock test")
        else:
            # Fallback if mock doesn't have force_game_over (it should now)
            dummy_state.step_count = 10  # Assuming step_count >= 5 makes it terminal

    assert dummy_state.is_over()

    print("\n--- Starting MCTS Run on Terminal State ---")
    visit_counts = run_mcts(dummy_state, dummy_network, search_config)
    print(f"Visit Counts: {visit_counts}")

    assert isinstance(visit_counts, dict)
    assert len(visit_counts) == 0  # MCTS should return empty for terminal root

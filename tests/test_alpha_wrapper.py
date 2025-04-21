# File: tests/test_alpha_wrapper.py
import time
from typing import Any, TypeAlias  # Import TypeAlias

import pytest
import numpy as np  # Import numpy

# --- Define GameState and EnvConfig types ---
# Try importing the real ones first
try:
    # Use absolute imports
    from trianglengin import EnvConfig as RealEnvConfig
    from trianglengin import GameState as RealGameState

    HAS_TRIANGLENGIN = True
    # Define type aliases pointing to the real classes
    GameState: TypeAlias = RealGameState
    EnvConfig: TypeAlias = RealEnvConfig
except ImportError:
    HAS_TRIANGLENGIN = False
    print("WARNING: 'trianglengin' not found. Using mock GameState for tests.")

    # Define minimal mock classes if trianglengin is not available
    class MockGameStateForTest:
        def __init__(self, *_args: Any, **_kwargs: Any):  # Prefix unused args
            self.step_count = 0
            self._is_over = False  # Add state for forcing terminal
            self.current_step = 0  # Add current_step for logging in wrapper
            # Make mock more deterministic for testing batch counts
            self._valid_actions = [0, 1]  # Start with some actions

        def copy(self) -> "MockGameStateForTest":
            s = MockGameStateForTest()
            s.step_count = self.step_count
            s._is_over = self._is_over
            s.current_step = self.current_step
            s._valid_actions = list(self._valid_actions)  # Copy list
            return s

        # Make step deterministic for testing
        def step(self, action: int) -> tuple[float, bool]:
            if self._is_over:
                return 0.0, True
            self.step_count += 1
            self.current_step += 1
            # Deterministic terminal condition
            if self.step_count >= 5:
                self._is_over = True
                self._valid_actions = []
            # Deterministic action validity change (e.g., action 1 becomes invalid after 2 steps)
            elif self.step_count >= 2 and 1 in self._valid_actions:
                self._valid_actions.remove(1)
            else:
                self._valid_actions = [0, 1]  # Ensure it resets if not terminal

            return 0.0, self._is_over

        def is_over(self) -> bool:
            return self._is_over or not self._valid_actions

        def get_outcome(self) -> float:
            return -1.0 if self._is_over and self.step_count >= 5 else 0.0

        def valid_actions(self) -> list[int]:  # Changed mock to return list
            return list(self._valid_actions)  # Return copy

        def force_game_over(self, reason: str) -> None:
            print(f"Mock force_game_over: {reason}")
            self._is_over = True
            self._valid_actions = []

    class MockEnvConfigForTest:
        pass

    # Define type aliases pointing to the mock classes
    GameState: TypeAlias = MockGameStateForTest  # type: ignore
    EnvConfig: TypeAlias = MockEnvConfigForTest  # type: ignore
# --- End GameState/EnvConfig Definition ---


# Now import trimcts components AFTER GameState/EnvConfig are defined
from trimcts import AlphaZeroNetworkInterface, SearchConfiguration, run_mcts


# --- Dummy Network Implementation ---
class DummyAlphaNetwork(AlphaZeroNetworkInterface):
    """A simple network that returns fixed policy/value."""

    def __init__(self, action_dim: int = 2, value: float = 0.5, delay: float = 0.0):
        self.action_dim = action_dim
        self.value = value
        # Ensure policy covers potential actions
        self.policy = {i: 1.0 / action_dim for i in range(action_dim)}
        self.eval_count = 0
        self.batch_eval_count = 0
        self.total_states_evaluated = 0  # Add counter for total states
        self.delay = delay

    def reset_counters(self) -> None:
        self.eval_count = 0
        self.batch_eval_count = 0
        self.total_states_evaluated = 0

    def evaluate_state(self, state: GameState) -> tuple[dict[int, float], float]:
        self.eval_count += 1
        self.total_states_evaluated += 1
        if self.delay > 0:
            time.sleep(self.delay)

        valid_actions = state.valid_actions()
        if not valid_actions:
            return {}, self.value

        # Use the stored policy dimension, filter by valid actions
        valid_policy = {a: p for a, p in self.policy.items() if a in valid_actions}
        policy_sum = sum(valid_policy.values())

        if policy_sum > 1e-6:
            normalized_policy = {a: p / policy_sum for a, p in valid_policy.items()}
        elif valid_actions:  # Handle case where valid actions exist but policy was zero
            uniform_prob = 1.0 / len(valid_actions)
            normalized_policy = dict.fromkeys(valid_actions, uniform_prob)
        else:  # No valid actions, empty policy
            normalized_policy = {}

        return normalized_policy, self.value

    def evaluate_batch(
        self, states: list[GameState]
    ) -> list[tuple[dict[int, float], float]]:
        self.batch_eval_count += 1
        if self.delay > 0:
            time.sleep(self.delay * 0.1 * len(states))
        results = [self.evaluate_state(s) for s in states]
        return results


# --- Test Fixtures ---
@pytest.fixture
def dummy_state() -> GameState:
    """Provides a simple dummy game state."""
    if HAS_TRIANGLENGIN:
        test_playable_range = [(0, 3), (0, 3), (0, 3)]
        test_config = RealEnvConfig(
            ROWS=3,
            COLS=3,
            NUM_SHAPE_SLOTS=1,
            PLAYABLE_RANGE_PER_ROW=test_playable_range,
        )
        return RealGameState(
            config=test_config, initial_seed=np.random.randint(0, 10000)
        )
    else:
        return GameState()


@pytest.fixture
def dummy_network() -> DummyAlphaNetwork:
    """Provides a dummy network interface, resetting counters each time."""
    # Use action_dim=3 for mock to match trianglengin 3x3 default better
    action_dim = 3 * 3 * 1 if HAS_TRIANGLENGIN else 2
    net = DummyAlphaNetwork(action_dim=action_dim, value=0.1)
    return net


@pytest.fixture
def search_config() -> SearchConfiguration:
    """Provides a default search configuration."""
    return SearchConfiguration(
        max_simulations=16,
        max_depth=5,
        cpuct=1.25,
        dirichlet_alpha=0.3,
        dirichlet_epsilon=0.25,
        discount=1.0,
        mcts_batch_size=1,  # Default to 1
    )


# --- Tests ---


def test_mcts_run_alpha_basic(
    dummy_state: GameState,
    dummy_network: DummyAlphaNetwork,
    search_config: SearchConfiguration,
) -> None:
    """Test basic MCTS run with AlphaZero interface (batch_size=1)."""
    num_sims = 10
    search_config.max_simulations = num_sims
    search_config.dirichlet_alpha = 0.0
    search_config.mcts_batch_size = 1

    if dummy_state.is_over():
        pytest.skip("Initial dummy state is already terminal.")

    print("\n--- Starting Basic MCTS Run (Batch Size 1) ---")
    start_time = time.time()
    visit_counts = run_mcts(dummy_state, dummy_network, search_config)
    duration = time.time() - start_time
    print(f"--- MCTS Run Finished ({duration:.4f}s) ---")
    print(f"Visit Counts: {visit_counts}")
    print(f"Network single evals (evaluate_state calls): {dummy_network.eval_count}")
    print(
        f"Network batch evals (evaluate_batch calls): {dummy_network.batch_eval_count}"
    )
    print(f"Total states evaluated: {dummy_network.total_states_evaluated}")

    assert isinstance(visit_counts, dict)
    valid_root_actions = set(dummy_state.valid_actions())
    if valid_root_actions:
        assert set(visit_counts.keys()).issubset(valid_root_actions)
        assert all(isinstance(v, int) and v >= 0 for v in visit_counts.values())
        assert sum(visit_counts.values()) <= num_sims
        assert len(visit_counts) > 0
        # With batch_size=1, evaluate_batch is called for root + each leaf found
        assert dummy_network.batch_eval_count >= 1
        assert dummy_network.batch_eval_count <= num_sims + 1
        # Total states evaluated should equal batch calls when batch_size=1
        assert dummy_network.total_states_evaluated == dummy_network.batch_eval_count
    else:
        assert not visit_counts


def test_mcts_run_alpha_with_noise(
    dummy_state: GameState,
    dummy_network: DummyAlphaNetwork,
    search_config: SearchConfiguration,
) -> None:
    """Test MCTS run with Dirichlet noise enabled (batch_size=1)."""
    num_sims = 32
    search_config.max_simulations = num_sims
    search_config.dirichlet_alpha = 0.5
    search_config.dirichlet_epsilon = 0.25
    search_config.mcts_batch_size = 1

    if dummy_state.is_over():
        pytest.skip("Initial dummy state is already terminal.")

    print("\n--- Starting MCTS Run with Noise (Batch Size 1) ---")
    start_time = time.time()
    visit_counts = run_mcts(dummy_state, dummy_network, search_config)
    duration = time.time() - start_time
    print(f"--- MCTS Run Finished ({duration:.4f}s) ---")
    print(f"Visit Counts: {visit_counts}")
    print(f"Network single evals (evaluate_state calls): {dummy_network.eval_count}")
    print(
        f"Network batch evals (evaluate_batch calls): {dummy_network.batch_eval_count}"
    )
    print(f"Total states evaluated: {dummy_network.total_states_evaluated}")

    assert isinstance(visit_counts, dict)
    valid_root_actions = set(dummy_state.valid_actions())
    if valid_root_actions:
        assert set(visit_counts.keys()).issubset(valid_root_actions)
        assert all(isinstance(v, int) and v >= 0 for v in visit_counts.values())
        assert sum(visit_counts.values()) <= num_sims
        assert dummy_network.batch_eval_count >= 1
        assert dummy_network.batch_eval_count <= num_sims + 1
        assert dummy_network.total_states_evaluated == dummy_network.batch_eval_count
    else:
        assert not visit_counts


def test_mcts_run_on_terminal_state(
    dummy_state: GameState,
    dummy_network: DummyAlphaNetwork,
    search_config: SearchConfiguration,
) -> None:
    """Test MCTS run starting from a terminal state (should return empty)."""
    steps = 0
    max_steps = 100
    while not dummy_state.is_over() and steps < max_steps:
        actions_set = set(dummy_state.valid_actions())
        if not actions_set:
            # If no actions, force game over manually for mock, or break for real
            if hasattr(dummy_state, "force_game_over"):
                dummy_state.force_game_over("No actions left in test setup")
            break
        actions_list = list(actions_set)
        # Use try-except for choice as it can fail if list is empty after check (race?)
        try:
            action = np.random.choice(actions_list) if actions_list else -1
        except ValueError:
            action = -1  # Handle empty list case if it occurs

        if action == -1:
            break
        dummy_state.step(action)
        steps += 1

    if not dummy_state.is_over():
        # If using real game, it might be possible to not terminate, skip then.
        if HAS_TRIANGLENGIN:
            pytest.skip(
                "Real game state did not terminate naturally within test limits."
            )
        else:  # Mock should always terminate
            pytest.fail("Mock game state did not terminate as expected.")

    assert dummy_state.is_over()

    print("\n--- Starting MCTS Run on Terminal State ---")
    visit_counts = run_mcts(dummy_state, dummy_network, search_config)
    print(f"Visit Counts: {visit_counts}")

    assert isinstance(visit_counts, dict)
    assert len(visit_counts) == 0
    assert dummy_network.total_states_evaluated == 0
    assert dummy_network.batch_eval_count == 0


# --- New Tests for Batching ---


def test_mcts_batching_enabled(
    dummy_state: GameState,
    dummy_network: DummyAlphaNetwork,
    search_config: SearchConfiguration,
) -> None:
    """Test MCTS run with batching enabled (batch_size > 1)."""
    batch_size = 4
    num_sims = 20
    search_config.max_simulations = num_sims
    search_config.mcts_batch_size = batch_size
    search_config.dirichlet_alpha = 0.0

    if dummy_state.is_over():
        pytest.skip("Initial dummy state is already terminal.")

    print(f"\n--- Starting MCTS Run (Batch Size {batch_size}) ---")
    start_time = time.time()
    visit_counts = run_mcts(dummy_state, dummy_network, search_config)
    duration = time.time() - start_time
    print(f"--- MCTS Run Finished ({duration:.4f}s) ---")
    print(f"Visit Counts: {visit_counts}")
    print(f"Network single evals (evaluate_state calls): {dummy_network.eval_count}")
    print(
        f"Network batch evals (evaluate_batch calls): {dummy_network.batch_eval_count}"
    )
    print(f"Total states evaluated: {dummy_network.total_states_evaluated}")

    assert isinstance(visit_counts, dict)
    valid_root_actions = set(dummy_state.valid_actions())
    if valid_root_actions:
        assert set(visit_counts.keys()).issubset(valid_root_actions)
        assert sum(visit_counts.values()) <= num_sims

        # --- Relaxed Assertions for Batching ---
        assert dummy_network.batch_eval_count >= 1  # Must eval root
        assert dummy_network.total_states_evaluated >= 1  # Must eval root
        # If sims > 1 and leaves were found, expect > 1 batch call
        if num_sims > 1 and dummy_network.total_states_evaluated > 1:
            assert dummy_network.batch_eval_count > 1, (
                "Batching didn't occur for leaves"
            )
        # Upper bound check
        assert dummy_network.batch_eval_count <= num_sims + 1
        assert dummy_network.total_states_evaluated <= num_sims + 1
        # Check internal consistency of mock network counters
        assert dummy_network.eval_count == dummy_network.total_states_evaluated

    else:
        assert not visit_counts
        assert dummy_network.total_states_evaluated == 0
        assert dummy_network.batch_eval_count == 0


def test_mcts_batching_small_sims(
    dummy_state: GameState,
    dummy_network: DummyAlphaNetwork,
    search_config: SearchConfiguration,
) -> None:
    """Test MCTS run with batching enabled but fewer sims than batch size."""
    batch_size = 8
    num_sims = 5
    search_config.max_simulations = num_sims
    search_config.mcts_batch_size = batch_size
    search_config.dirichlet_alpha = 0.0

    if dummy_state.is_over():
        pytest.skip("Initial dummy state is already terminal.")

    print(f"\n--- Starting MCTS Run (Sims={num_sims}, Batch Size={batch_size}) ---")
    start_time = time.time()
    visit_counts = run_mcts(dummy_state, dummy_network, search_config)
    duration = time.time() - start_time
    print(f"--- MCTS Run Finished ({duration:.4f}s) ---")
    print(f"Visit Counts: {visit_counts}")
    print(f"Network single evals (evaluate_state calls): {dummy_network.eval_count}")
    print(
        f"Network batch evals (evaluate_batch calls): {dummy_network.batch_eval_count}"
    )
    print(f"Total states evaluated: {dummy_network.total_states_evaluated}")

    assert isinstance(visit_counts, dict)
    valid_root_actions = set(dummy_state.valid_actions())
    if valid_root_actions:
        assert set(visit_counts.keys()).issubset(valid_root_actions)
        assert sum(visit_counts.values()) <= num_sims

        # --- Relaxed Assertions for Batching ---
        assert dummy_network.batch_eval_count >= 1  # Must eval root
        assert dummy_network.total_states_evaluated >= 1  # Must eval root
        # Expect 1 (root) + maybe 1 (leaves) = 1 or 2 calls total
        assert dummy_network.batch_eval_count <= 2, (
            "Expected at most 2 batch calls (root + leaves)"
        )
        # Upper bound check
        assert dummy_network.batch_eval_count <= num_sims + 1
        assert dummy_network.total_states_evaluated <= num_sims + 1
        # Check internal consistency of mock network counters
        assert dummy_network.eval_count == dummy_network.total_states_evaluated

    else:
        assert not visit_counts
        assert dummy_network.total_states_evaluated == 0
        assert dummy_network.batch_eval_count == 0

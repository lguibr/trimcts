# File: trimcts/src/trimcts/mcts_wrapper.py
import logging
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from .config import SearchConfiguration

logger = logging.getLogger(__name__)

# Type hint for the game state object expected by the network interfaces
GameState = Any
# Type hint for the opaque tree handle (using Any for now)
MctsTreeHandle = Any

# --- Conditional Import for MyPy ---
if TYPE_CHECKING:
    from . import trimcts_cpp as trimcts_cpp_stub

    trimcts_cpp: type[trimcts_cpp_stub]
# --- End Conditional Import ---


@runtime_checkable
class AlphaZeroNetworkInterface(Protocol):
    def evaluate_state(self, state: GameState) -> tuple[dict[int, float], float]: ...

    def evaluate_batch(
        self, states: list[GameState]
    ) -> list[tuple[dict[int, float], float]]: ...


@runtime_checkable
class MuZeroNetworkInterface(Protocol):
    def initial_inference(
        self, state: GameState
    ) -> tuple[dict[int, float], float, Any]: ...

    def recurrent_inference(
        self, hidden_state: Any, action: int
    ) -> tuple[dict[int, float], float, Any]: ...


def run_mcts(
    root_state: GameState,
    network_interface: AlphaZeroNetworkInterface | MuZeroNetworkInterface,
    config: SearchConfiguration,
    previous_tree_handle: MctsTreeHandle | None = None,
    last_action: int = -1,
) -> tuple[dict[int, int], MctsTreeHandle | None, float]:
    """
    Python entry point for running MCTS, supporting tree reuse.

    Args:
        root_state: The current game state object.
        network_interface: The network evaluation interface.
        config: The MCTS search configuration.
        previous_tree_handle: Opaque handle to the tree state from the previous step (optional).
        last_action: The action taken that led to the current root_state (required if previous_tree_handle is provided).

    Returns:
        A tuple containing:
            - Visit counts (dict[int, int]) for actions from the root.
            - An opaque handle to the MCTS tree state after the search (or None if MCTS failed).
            - The average depth reached across all simulations performed (float).
            Returns ({}, None, 0.0) immediately if root_state.is_over() is True.
    """
    # Terminal-state shortcut
    if not hasattr(root_state, "is_over") or not callable(root_state.is_over):
        raise TypeError("root_state object missing required method: is_over")
    if root_state.is_over():
        logger.warning("run_mcts called on a terminal state. Returning empty.")
        return {}, None, 0.0

    # Validate config
    if not isinstance(config, SearchConfiguration):
        raise TypeError("config must be an instance of SearchConfiguration")

    # Validate tree reuse parameters
    if previous_tree_handle is not None and last_action == -1:
        logger.warning(
            "previous_tree_handle provided but last_action is -1. Tree reuse might fail. "
            "Provide the action that led to the current root_state."
        )
    if previous_tree_handle is None and last_action != -1:
        logger.debug(
            "last_action provided but no previous_tree_handle. Starting new tree."
        )
        last_action = -1  # Ensure last_action is -1 if no handle

    # Import the C++ extension
    try:
        import trimcts.trimcts_cpp as cpp_module
    except ImportError as e:
        raise ImportError(
            "TriMCTS C++ extension module ('trimcts.trimcts_cpp') not found or failed to import. "
            "Ensure the package was built correctly (`pip install -e .`). "
            f"Original error: {e}"
        ) from e

    # Ensure expected function exists
    if not hasattr(cpp_module, "run_mcts_cpp"):
        raise RuntimeError(
            "Loaded module missing 'run_mcts_cpp'. Build might be incomplete or corrupted."
        )

    # Validate root_state capabilities
    for method in ("copy", "step", "get_outcome", "valid_actions"):
        if not hasattr(root_state, method) or not callable(getattr(root_state, method)):
            raise TypeError(f"root_state object missing required method: {method}")

    # Network interface type check
    is_alpha = isinstance(network_interface, AlphaZeroNetworkInterface)
    is_mu = isinstance(network_interface, MuZeroNetworkInterface)
    if not is_alpha and not is_mu:
        raise TypeError(
            "network_interface must implement AlphaZeroNetworkInterface or MuZeroNetworkInterface"
        )
    if is_alpha and is_mu:
        logger.warning(
            "network_interface implements both AlphaZero and MuZero. Assuming AlphaZero."
        )
        is_mu = False

    if is_mu:
        raise NotImplementedError(
            "MuZero MCTS integration is not yet implemented in C++ bindings."
        )

    # Call into C++ - it now returns a tuple (visit_map, new_handle, avg_depth)
    try:
        # Pass None directly if previous_tree_handle is None
        handle_to_pass = (
            previous_tree_handle if previous_tree_handle is not None else None
        )
        result_tuple = cpp_module.run_mcts_cpp(
            root_state,
            network_interface,
            config,
            handle_to_pass,  # Pass handle (can be None)
            last_action,
        )
    except Exception as cpp_err:
        logger.error(f"Error during C++ MCTS execution: {cpp_err}", exc_info=True)
        return {}, None, 0.0  # Return empty on C++ error

    # Validate and unpack the returned tuple
    if not isinstance(result_tuple, tuple) or len(result_tuple) != 3:
        logger.error(
            f"C++ MCTS returned unexpected type or length: {type(result_tuple)}"
        )
        return {}, None, 0.0

    visit_counts_raw, new_tree_handle, avg_depth_raw = result_tuple

    # Validate visit counts
    validated_visit_counts: dict[int, int] = {}  # Use a different name here
    if not isinstance(visit_counts_raw, dict):
        logger.error(
            f"C++ MCTS returned unexpected type for visit counts: {type(visit_counts_raw)}"
        )
        # validated_visit_counts remains empty
    else:
        # Filter and validate keys/values
        for k, v in visit_counts_raw.items():
            if isinstance(k, int) and isinstance(v, int):
                validated_visit_counts[k] = v  # Populate the new dict
            else:
                logger.warning(
                    f"Skipping invalid result entry: ({k!r}:{type(k)}, {v!r}:{type(v)})"
                )

    # Validate average depth
    avg_depth: float = 0.0
    if isinstance(avg_depth_raw, float | int):
        avg_depth = float(avg_depth_raw)
    else:
        logger.error(
            f"C++ MCTS returned unexpected type for average depth: {type(avg_depth_raw)}"
        )

    # The handle can be None if C++ returns a null capsule (e.g., on error or terminal state)
    # Check if the returned handle is None (Python None, not a null capsule object)
    if new_tree_handle is None:
        logger.debug("C++ MCTS returned None for the tree handle.")
        return validated_visit_counts, None, avg_depth

    # If it's not None, assume it's a valid capsule handle
    return validated_visit_counts, new_tree_handle, avg_depth

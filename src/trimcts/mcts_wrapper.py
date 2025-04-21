import logging
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable, cast

from .config import SearchConfiguration

logger = logging.getLogger(__name__)

# Type hint for the game state object expected by the network interfaces
GameState = Any

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
) -> dict[int, int]:
    """
    Python entry point for running MCTS.

    Returns empty dict immediately if root_state.is_over() is True.
    """
    # Terminal-state shortcut
    if not hasattr(root_state, "is_over") or not callable(root_state.is_over):
        raise TypeError("root_state object missing required method: is_over")
    if root_state.is_over():
        return {}

    # Validate config
    if not isinstance(config, SearchConfiguration):
        raise TypeError("config must be an instance of SearchConfiguration")

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

    # Call into C++
    visit_counts = cast(
        dict[int, int],
        cpp_module.run_mcts_cpp(root_state, network_interface, config),
    )

    # Validate return type
    if not isinstance(visit_counts, dict):
        logger.error(f"C++ MCTS returned unexpected type: {type(visit_counts)}")
        return {}

    # Filter and validate keys/values
    result: dict[int, int] = {}
    for k, v in visit_counts.items():
        if isinstance(k, int) and isinstance(v, int):
            result[k] = v
        else:
            logger.warning(
                f"Skipping invalid result entry: ({k!r}:{type(k)}, {v!r}:{type(v)})"
            )
    return result

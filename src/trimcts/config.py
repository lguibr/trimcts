# File: src/trimcts/config.py
"""
Python configuration class for MCTS parameters.
Uses Pydantic for validation.
"""

from pydantic import BaseModel, ConfigDict, Field  # Import ConfigDict


class SearchConfiguration(BaseModel):
    """MCTS Search Configuration."""

    # Core Search Parameters
    max_simulations: int = Field(
        default=50, description="Maximum number of MCTS simulations per move.", gt=0
    )
    max_depth: int = Field(
        default=10, description="Maximum depth for tree traversal.", gt=0
    )

    # UCT Parameters (AlphaZero style)
    cpuct: float = Field(
        default=1.25,
        description="Constant determining the level of exploration (PUCT).",
    )

    # Dirichlet Noise (for root node exploration)
    dirichlet_alpha: float = Field(
        default=0.3, description="Alpha parameter for Dirichlet noise.", ge=0
    )
    dirichlet_epsilon: float = Field(
        default=0.25,
        description="Weight of Dirichlet noise in root prior probabilities.",
        ge=0,
        le=1.0,
    )

    # Discount Factor (Primarily for MuZero/Value Propagation)
    discount: float = Field(
        default=1.0,
        description="Discount factor (gamma) for future rewards/values.",
        ge=0.0,
        le=1.0,
    )

    # Temperature (for action selection after search)
    # Note: Temperature for selection *during* search is often implicitly 1.0
    # This might be applied *after* getting visit counts from C++.
    # We won't pass this to C++ for now, apply it in Python if needed.
    # selection_temperature: float = Field(default=1.0, ge=0)

    # Batching (Handled by C++ internally for now)
    # virtual_loss: float = Field(default=1.0) # Common technique, add later if needed
    # num_parallel_traversals: int = Field(default=8) # Add later if needed

    # MuZero Specific (Add later)
    # value_delta_max: float = Field(default=0.01) # MuZero specific

    # Use ConfigDict for Pydantic V2
    model_config = ConfigDict(validate_assignment=True)

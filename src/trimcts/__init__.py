"""
TriMCTS Package

Provides high-performance C++ MCTS bindings for Python, supporting tree reuse.
"""

# Import only Python-defined elements here
from .config import SearchConfiguration
from .mcts_wrapper import AlphaZeroNetworkInterface, MuZeroNetworkInterface, run_mcts

__all__ = [
    "run_mcts",
    "SearchConfiguration",
    "AlphaZeroNetworkInterface",
    "MuZeroNetworkInterface",
]

# Increment version for tree reuse feature
__version__ = "1.2.0"

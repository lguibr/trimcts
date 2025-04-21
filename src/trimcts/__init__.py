# File: src/trimcts/__init__.py
"""
TriMCTS Package

Provides high-performance C++ MCTS bindings for Python.
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

__version__ = "0.1.0"

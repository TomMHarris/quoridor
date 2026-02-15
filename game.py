"""
Game engine auto-selector.

Usage in other files:
    from game import QuoridorGame, MoveEncoder

Automatically uses the fast Cython engine if compiled,
otherwise falls back to the pure Python engine.
"""

try:
    from engine_fast import FastQuoridorGame as _FastGame
    ENGINE = "cython"

    class QuoridorGame(_FastGame):
        """Thin wrapper adding class constants for API compatibility."""
        BOARD_SIZE = 9
        WALL_GRID = 8

    print("[engine] Using Cython engine (fast)")
except ImportError:
    from engine import QuoridorGame
    ENGINE = "python"
    print("[engine] Using pure Python engine (slow)")

# MoveEncoder is pure Python (lightweight, no hot path)
from engine import MoveEncoder, Move

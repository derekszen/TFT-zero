"""Compatibility exports for the core state module."""

from mini_tft.core.ids import EMPTY
from mini_tft.core.state import GameState, UnitInstance, new_game_state, state_signature

__all__ = ["EMPTY", "GameState", "UnitInstance", "new_game_state", "state_signature"]

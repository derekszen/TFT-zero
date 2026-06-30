"""Canonical strategic simulator core."""

from mini_tft.strategic.core.actions import NUM_ACTIONS, StrategicAction
from mini_tft.strategic.core.obs import observation_dim, observe
from mini_tft.strategic.core.rules import (
    board_strength,
    enemy_strength_for_round,
    field_strongest,
    legal_action_mask,
    placement_proxy,
    reset,
    scenario_score,
    stage_label_for_round,
    stage_round_for_round,
    step,
)
from mini_tft.strategic.core.state import (
    StrategicConfig,
    StrategicState,
    state_signature,
)

__all__ = [
    "NUM_ACTIONS",
    "StrategicAction",
    "StrategicConfig",
    "StrategicState",
    "board_strength",
    "enemy_strength_for_round",
    "field_strongest",
    "legal_action_mask",
    "observe",
    "observation_dim",
    "placement_proxy",
    "reset",
    "scenario_score",
    "stage_label_for_round",
    "stage_round_for_round",
    "state_signature",
    "step",
]

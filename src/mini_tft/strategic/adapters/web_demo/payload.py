"""Serializable payloads for a future strategic playable demo."""

from __future__ import annotations

from typing import Any

from mini_tft.strategic.core.actions import action_name
from mini_tft.strategic.core.rules import (
    board_strength,
    enemy_strength_for_round,
    legal_action_mask,
    placement_proxy,
    scenario_score,
    stage_label_for_round,
    stage_round_for_round,
)
from mini_tft.strategic.core.state import DEFAULT_STRATEGIC_CONFIG, StrategicConfig, StrategicState


def state_payload(
    state: StrategicState,
    config: StrategicConfig = DEFAULT_STRATEGIC_CONFIG,
) -> dict[str, Any]:
    mask = legal_action_mask(state, config)
    stage, stage_round = stage_round_for_round(state.round)
    return {
        "round": state.round,
        "stage": stage,
        "stage_round": stage_round,
        "stage_label": stage_label_for_round(state.round),
        "hp": state.hp,
        "gold": state.gold,
        "level": state.level,
        "xp": state.xp,
        "shop": [int(value) for value in state.shop.tolist()],
        "owned": [int(value) for value in state.owned.tolist()],
        "fielded": [int(value) for value in state.fielded.tolist()],
        "role_items": [int(value) for value in state.role_items.tolist()],
        "role_item_slots": [int(value) for value in state.role_item_slots.tolist()],
        "board_strength": board_strength(state, config),
        "enemy_strength": enemy_strength_for_round(state.round, config),
        "placement_proxy": placement_proxy(state, config),
        "scenario_score": scenario_score(state, config),
        "legal_actions": [
            {"id": int(index), "name": action_name(int(index))}
            for index, legal in enumerate(mask)
            if bool(legal)
        ],
        "done": state.done,
        "final_reason": state.final_reason,
    }

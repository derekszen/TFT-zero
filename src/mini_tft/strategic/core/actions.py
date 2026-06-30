"""Strategic action contract for the clean MiniTFT lane."""

from __future__ import annotations

from enum import IntEnum


class StrategicAction(IntEnum):
    HOLD = 0
    LEVEL = 1
    ROLL = 2
    BUY_BEST_UPGRADE = 3
    BUY_BEST_SYNERGY = 4
    BUY_HIGHEST_COST = 5
    FIELD_STRONGEST = 6
    GREED_ECON = 7
    SLAM_CARRY_ITEM = 8
    SLAM_TANK_ITEM = 9
    SLAM_SUPPORT_ITEM = 10


NUM_ACTIONS = len(StrategicAction)

SLAM_ACTION_BY_ROLE = {
    "carry": StrategicAction.SLAM_CARRY_ITEM,
    "tank": StrategicAction.SLAM_TANK_ITEM,
    "support": StrategicAction.SLAM_SUPPORT_ITEM,
}
ROLE_BY_SLAM_ACTION = {action: role for role, action in SLAM_ACTION_BY_ROLE.items()}
ROLE_INDEX = {"carry": 0, "tank": 1, "support": 2}
ROLE_NAMES = ("carry", "tank", "support")


def action_name(action: int) -> str:
    try:
        return StrategicAction(action).name.lower()
    except ValueError:
        return "unknown"

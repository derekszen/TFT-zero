"""Set-1-like round schedule helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

RoundType = Literal["pve", "pvp"]


@dataclass(frozen=True)
class RoundInfo:
    round_num: int
    stage: int
    stage_round: int
    stage_label: str
    round_type: RoundType

    @property
    def is_pve(self) -> bool:
        return self.round_type == "pve"


def round_info(round_num: int) -> RoundInfo:
    """Map the simulator's 1-indexed round counter to TFT-style stage metadata."""

    if round_num < 1:
        raise ValueError(f"round_num must be positive: {round_num}")

    if round_num <= 3:
        stage = 1
        stage_round = round_num
    else:
        offset = round_num - 4
        stage = 2 + offset // 7
        stage_round = 1 + offset % 7

    round_type: RoundType = "pve" if stage == 1 or stage_round == 7 else "pvp"
    return RoundInfo(
        round_num=round_num,
        stage=stage,
        stage_round=stage_round,
        stage_label=f"Stage {stage}-{stage_round}",
        round_type=round_type,
    )


def is_item_drop_round(round_num: int) -> bool:
    """Return whether the round should award a simple PvE item drop."""

    return round_info(round_num).is_pve

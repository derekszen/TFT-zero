"""Configuration for the V0 simulator."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EnvConfig:
    """Knobs for the single-player Set-1-like simulator."""

    dataset: str = "set1"
    seed: int | None = None
    max_round: int = 36
    max_steps_per_episode: int = 1_000
    starting_hp: int = 100
    starting_gold: int = 0
    starting_level: int = 3
    starting_xp: int = 0
    max_level: int = 9
    shop_size: int = 5
    bench_size: int = 9
    item_bench_size: int = 10
    max_items_on_unit: int = 3
    roll_cost: int = 2
    xp_buy_cost: int = 4
    xp_per_buy: int = 4
    base_income: int = 5
    max_interest: int = 5
    win_gold: int = 1
    combat_noise_std: float = 3.0
    combat_sigmoid_scale: float = 12.0
    item_drop_interval: int = 4

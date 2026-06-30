from collections.abc import Sequence
from typing import Any

def trace_script(
    *,
    seed: int,
    actions: Sequence[int],
    max_round: int = ...,
    max_actions_per_round: int = ...,
) -> list[tuple[object, ...]]: ...

def run_mcts_smoke(
    *,
    episodes: int,
    seed: int,
    simulations: Sequence[int],
    max_depth: int,
    rollout_steps: int,
    prior_mode: str = ...,
) -> dict[str, Any]: ...

def plan_batch_from_seeds(
    *,
    seeds: Sequence[int],
    simulations: int,
    max_depth: int,
    rollout_steps: int,
    prior_mode: str = ...,
) -> dict[str, Any]: ...

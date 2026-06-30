"""Python wrapper for the compiled simulator-backed strategic MCTS backend."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

try:
    from mini_tft.strategic.native import (  # type: ignore[reportMissingImports]
        _native as _native_module,
    )
except ImportError as exc:  # pragma: no cover - exercised only when the extension is absent.
    _native_module: Any | None = None
    _IMPORT_ERROR: ImportError | None = exc
else:
    _IMPORT_ERROR = None


def native_available() -> bool:
    return _native_module is not None


def trace_signatures(
    *,
    seed: int,
    actions: Sequence[int],
    max_round: int = 36,
    max_actions_per_round: int = 3,
) -> list[tuple[object, ...]]:
    native = _require_native()
    return list(
        native.trace_script(
            seed=seed,
            actions=list(actions),
            max_round=max_round,
            max_actions_per_round=max_actions_per_round,
        )
    )


def run_native_mcts_smoke(
    *,
    episodes: int,
    seed: int,
    simulations: Sequence[int],
    max_depth: int,
    rollout_steps: int,
    prior_mode: str,
) -> dict[str, Any]:
    native = _require_native()
    return dict(
        native.run_mcts_smoke(
            episodes=episodes,
            seed=seed,
            simulations=list(simulations),
            max_depth=max_depth,
            rollout_steps=rollout_steps,
            prior_mode=prior_mode,
        )
    )


def plan_batch(
    *,
    seeds: Sequence[int],
    simulations: int,
    max_depth: int,
    rollout_steps: int,
    prior_mode: str = "uniform",
) -> dict[str, Any]:
    native = _require_native()
    return dict(
        native.plan_batch_from_seeds(
            seeds=list(seeds),
            simulations=simulations,
            max_depth=max_depth,
            rollout_steps=rollout_steps,
            prior_mode=prior_mode,
        )
    )


def _require_native() -> Any:
    if _native_module is None:
        raise RuntimeError("native strategic MCTS extension is not available") from _IMPORT_ERROR
    return _native_module

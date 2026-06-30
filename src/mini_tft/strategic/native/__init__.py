"""Compiled strategic-lane acceleration backend."""

from mini_tft.strategic.native.mcts import (
    native_available,
    plan_batch,
    run_native_mcts_smoke,
    trace_signatures,
)

__all__ = [
    "native_available",
    "plan_batch",
    "run_native_mcts_smoke",
    "trace_signatures",
]

"""Puffer or batched rollout adapter for strategic-lane throughput evidence."""

from mini_tft.strategic.adapters.puffer.benchmark import (
    BenchmarkResult,
    run_benchmark,
    write_benchmark_artifacts,
)

__all__ = ["BenchmarkResult", "run_benchmark", "write_benchmark_artifacts"]

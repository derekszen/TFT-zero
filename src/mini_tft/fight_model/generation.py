"""Multiprocessing fight-label generation."""

from __future__ import annotations

import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from mini_tft.fight_model.sampler import sample_staged_v1
from mini_tft.fight_model.schema import empty_arrays, encode_example
from mini_tft.fight_model.storage import ShardInfo, completed_shard_names, write_shard_atomic
from mini_tft.fight_model.teacher import make_teacher


@dataclass(frozen=True)
class GenerationReport:
    target_fights: int
    written_fights: int
    workers: int
    shard_size: int
    elapsed_sec: float
    fights_per_sec: float
    failures: int
    path: str


def generate_fight_labels(
    out: Path,
    target_fights: int,
    workers: int,
    shard_size: int,
    seed: int,
    teacher_name: str,
    teacher_root: str | None,
    sampling: str = "staged_v1",
    max_failure_rate: float = 0.03,
) -> GenerationReport:
    if sampling != "staged_v1":
        raise ValueError("only staged_v1 sampling is implemented")
    if target_fights <= 0:
        raise ValueError("target_fights must be positive")
    if shard_size <= 0:
        raise ValueError("shard_size must be positive")
    workers = max(1, workers)
    started = time.perf_counter()
    completed = completed_shard_names(out)
    jobs = _shard_jobs(target_fights, shard_size, seed)
    jobs = [job for job in jobs if job["shard_name"] not in completed]
    written = 0
    failures = 0

    if workers == 1:
        for job in jobs:
            info = _generate_shard_job(out, teacher_name, teacher_root, sampling, **job)
            written += info.count
            failures += info.failures
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [
                executor.submit(
                    _generate_shard_job,
                    out,
                    teacher_name,
                    teacher_root,
                    sampling,
                    **job,
                )
                for job in jobs
            ]
            for future in as_completed(futures):
                info = future.result()
                written += info.count
                failures += info.failures

    elapsed = time.perf_counter() - started
    if failures > max(1, int(max_failure_rate * max(1, written))):
        raise RuntimeError(
            f"fight-label failure count too high: {failures} failures for {written} fights"
        )
    return GenerationReport(
        target_fights=target_fights,
        written_fights=written,
        workers=workers,
        shard_size=shard_size,
        elapsed_sec=elapsed,
        fights_per_sec=written / elapsed if elapsed else 0.0,
        failures=failures,
        path=str(out),
    )


def _generate_shard_job(
    out: Path,
    teacher_name: str,
    teacher_root: str | None,
    sampling: str,
    shard_name: str,
    count: int,
    seed: int,
) -> ShardInfo:
    started = time.perf_counter()
    teacher = make_teacher(teacher_name, teacher_root)
    rng = np.random.default_rng(seed)
    arrays = empty_arrays(count)
    failures = 0
    row = 0
    attempts = 0
    while row < count:
        attempts += 1
        spec_seed = int(rng.integers(0, 2**63 - 1))
        spec = sample_staged_v1(
            rng,
            spec_seed,
            max_unit_id=teacher.max_unit_id,
            max_item_id=teacher.max_item_id,
        )
        try:
            label = teacher.simulate(spec)
        except Exception:
            failures += 1
            if attempts > count * 2:
                raise
            continue
        encode_example(arrays, row, spec, label)
        row += 1

    elapsed = time.perf_counter() - started
    return write_shard_atomic(
        out,
        shard_name,
        arrays,
        {
            "teacher": teacher.name,
            "teacher_requested": teacher_name,
            "sampling": sampling,
            "seed": seed,
            "elapsed_sec": elapsed,
            "failures": failures,
        },
    )


def _shard_jobs(target_fights: int, shard_size: int, seed: int) -> list[dict[str, Any]]:
    jobs = []
    start = 0
    shard_index = 0
    while start < target_fights:
        count = min(shard_size, target_fights - start)
        jobs.append(
            {
                "shard_name": f"shard_{shard_index:06d}",
                "count": count,
                "seed": seed + shard_index * 100_003,
            }
        )
        start += count
        shard_index += 1
    return jobs

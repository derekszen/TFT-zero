"""Benchmark parallel rollout worker counts."""

from __future__ import annotations

import argparse
from pathlib import Path
from tempfile import gettempdir

from mini_tft.tools.generate_bot_dataset import generate_dataset_parallel, resolve_worker_count


def parse_workers(value: str) -> list[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=1_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", default="1,2,4,8,16")
    args = parser.parse_args()

    rows = []
    for workers in parse_workers(args.workers):
        output = Path(gettempdir()) / f"tft_zero_rollout_tune_w{workers}.npz"
        metrics = generate_dataset_parallel(
            episodes=args.episodes,
            output=output,
            seed=args.seed,
            workers=workers,
        )
        rows.append(metrics)

    auto_workers = resolve_worker_count(0, args.episodes)
    best = max(rows, key=lambda row: float(row["transitions_per_sec"]))

    print("# MiniTFT Parallel Rollout Tuning")
    print()
    print(f"episodes: {args.episodes}")
    print(f"auto_workers: {auto_workers}")
    print(f"best_workers: {best['workers']}")
    print(f"best_transitions_per_sec: {best['transitions_per_sec']:.1f}")
    print()
    print("| Workers | Transitions | Seconds | Transitions/sec | File MB |")
    print("| ---: | ---: | ---: | ---: | ---: |")
    for row in rows:
        print(
            f"| {row['workers']} | {row['transitions']} | {row['elapsed_sec']:.3f} | "
            f"{row['transitions_per_sec']:.1f} | {row['file_mb']:.3f} |"
        )


if __name__ == "__main__":
    main()

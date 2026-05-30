"""Generate detailed fight-label datasets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mini_tft.fight_model.generation import generate_fight_labels


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("data/fight_labels/set4_teacher_v1"))
    parser.add_argument("--target-fights", type=int, default=1_000_000)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--shard-size", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--teacher",
        choices=["tft-muzero", "tft_muzero", "heuristic"],
        default="tft-muzero",
    )
    parser.add_argument("--teacher-root", type=str, default="third_party/TFTMuZeroAgent")
    parser.add_argument("--sampling", choices=["staged_v1"], default="staged_v1")
    parser.add_argument(
        "--max-failure-rate",
        type=float,
        default=0.03,
        help="Maximum resampled teacher failure rate before the job is treated as failed.",
    )
    args = parser.parse_args()

    report = generate_fight_labels(
        out=args.out,
        target_fights=args.target_fights,
        workers=args.workers,
        shard_size=args.shard_size,
        seed=args.seed,
        teacher_name=args.teacher,
        teacher_root=args.teacher_root,
        sampling=args.sampling,
        max_failure_rate=args.max_failure_rate,
    )
    print(json.dumps(report.__dict__, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

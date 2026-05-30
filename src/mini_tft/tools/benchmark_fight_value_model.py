"""Benchmark FightValueNet inference speed."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mini_tft.fight_model.benchmark import benchmark_model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=16_384)
    parser.add_argument("--batches", type=int, default=100)
    args = parser.parse_args()
    report = benchmark_model(args.checkpoint, args.device, args.batch_size, args.batches)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

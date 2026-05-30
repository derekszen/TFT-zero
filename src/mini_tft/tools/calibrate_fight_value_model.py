"""Compare a FightValueNet checkpoint against real comp ranking data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mini_tft.fight_model.calibration import (
    calibration_report,
    load_comp_strength_records,
    score_records_with_fight_value,
)
from mini_tft.fight_model.simulator_adapter import FightValueCombatModel

DEFAULT_FIXTURE = Path("tests/fixtures/metatft_set17_comp_strength_2026-05-31.json")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--min-pairwise-accuracy", type=float, default=0.65)
    parser.add_argument("--min-spearman", type=float, default=0.30)
    parser.add_argument("--min-top-k-overlap", type=float, default=0.50)
    parser.add_argument("--fail-on-threshold", action="store_true")
    args = parser.parse_args()

    source, records = load_comp_strength_records(args.fixture)
    evaluator = FightValueCombatModel(args.checkpoint, device_name=args.device)
    scores = score_records_with_fight_value(records, evaluator)
    report = calibration_report(
        records,
        scores,
        min_pairwise_accuracy=args.min_pairwise_accuracy,
        min_spearman=args.min_spearman,
        min_top_k_overlap=args.min_top_k_overlap,
    )
    payload = {
        "checkpoint": str(args.checkpoint),
        "fixture": str(args.fixture),
        "source": source,
        "scores": scores,
        "report": report.__dict__,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    if args.fail_on_threshold and not report.passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

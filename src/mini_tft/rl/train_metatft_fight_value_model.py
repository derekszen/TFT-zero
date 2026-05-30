"""Train FightValueNet from current-patch MetaTFT aggregate comp rankings."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from mini_tft.fight_model.calibration import load_comp_strength_records
from mini_tft.fight_model.metatft_training import (
    split_records_for_eval_exclusion,
    train_metatft_ranking_model,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-fixture", type=Path, required=True)
    parser.add_argument(
        "--eval-fixture",
        type=Path,
        default=Path("tests/fixtures/metatft_set17_comp_strength_2026-05-31.json"),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("checkpoints/fight_value/metatft_current_patch.pt"),
    )
    parser.add_argument("--device", default=_default_device())
    parser.add_argument("--epochs", type=int, default=4_000)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--embed-dim", type=int, default=16)
    parser.add_argument("--min-avg-gap", type=float, default=0.03)
    parser.add_argument(
        "--include-eval-clusters",
        action="store_true",
        help="Train on eval cluster IDs too. Use only for diagnostic overfit checks.",
    )
    args = parser.parse_args()

    train_source, train_records_all = load_comp_strength_records(args.train_fixture)
    eval_source, eval_records = load_comp_strength_records(args.eval_fixture)
    train_records = (
        train_records_all
        if args.include_eval_clusters
        else split_records_for_eval_exclusion(train_records_all, eval_records)
    )
    if not train_records:
        raise SystemExit("no train records remain after excluding eval clusters")

    report = train_metatft_ranking_model(
        train_records=train_records,
        eval_records=eval_records,
        output=args.out,
        device_name=args.device,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        seed=args.seed,
        hidden_dim=args.hidden_dim,
        embed_dim=args.embed_dim,
        min_avg_gap=args.min_avg_gap,
    )
    print(
        json.dumps(
            {
                **report.__dict__,
                "device": args.device,
                "train_fixture": str(args.train_fixture),
                "eval_fixture": str(args.eval_fixture),
                "train_source": train_source,
                "eval_source": eval_source,
                "records_train_available": len(train_records_all),
                "records_train_used": len(train_records),
                "eval_clusters_excluded": not args.include_eval_clusters,
            },
            indent=2,
            sort_keys=True,
        )
    )


def _default_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


if __name__ == "__main__":
    main()

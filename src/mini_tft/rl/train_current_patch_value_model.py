"""Train a current-patch MetaTFT board-value model from encoded catalog states."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from mini_tft.metatft import load_catalog_from_comp_strength
from mini_tft.metatft.value_training import train_current_patch_value_model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("checkpoints/fight_value/current_patch_board_value.pt"),
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--embed-dim", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--final-only", action="store_true")
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--min-avg-gap", type=float, default=0.05)
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument(
        "--blind-target-metadata",
        action="store_true",
        help="Zero target comp rank/stats and comp IDs to avoid heldout ranking leakage.",
    )
    args = parser.parse_args()

    catalog = load_catalog_from_comp_strength(args.catalog)
    report = train_current_patch_value_model(
        catalog,
        output=args.out,
        device_name=args.device,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        seed=args.seed,
        include_stage_lines=not args.final_only,
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        validation_fraction=args.validation_fraction,
        blind_target_metadata=args.blind_target_metadata,
        min_avg_gap=args.min_avg_gap,
        top_k=args.top_k,
    )
    print(json.dumps(report.__dict__, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

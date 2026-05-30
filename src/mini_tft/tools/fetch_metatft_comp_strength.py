"""Fetch a MetaTFT current-patch comp-strength snapshot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mini_tft.fight_model.metatft_data import (
    DEFAULT_RANKS,
    fetch_current_comp_strength,
    write_comp_strength_snapshot,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--queue", default="1100")
    parser.add_argument("--days", type=int, default=3)
    parser.add_argument("--ranks", default=",".join(DEFAULT_RANKS))
    parser.add_argument("--min-count", type=int, default=10_000)
    args = parser.parse_args()

    ranks = tuple(rank.strip() for rank in args.ranks.split(",") if rank.strip())
    payload = fetch_current_comp_strength(
        queue=args.queue,
        days=args.days,
        ranks=ranks,
        min_count=args.min_count,
    )
    write_comp_strength_snapshot(args.out, payload)
    print(
        json.dumps(
            {
                "path": str(args.out),
                "source": payload["source"],
                "records": len(payload["records"]),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

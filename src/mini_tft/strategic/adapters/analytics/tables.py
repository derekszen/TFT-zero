"""Artifact-table helpers for strategic-lane analytics."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

import polars as pl

EPISODE_SCHEMA = {
    "episode": pl.Int64,
    "policy": pl.Utf8,
    "placement": pl.Int64,
    "final_round": pl.Int64,
    "hp": pl.Int64,
    "final_reason": pl.Utf8,
    "scenario_score": pl.Float64,
    "illegal_actions": pl.Int64,
}


def episode_rows_frame(rows: Iterable[Mapping[str, Any]]) -> pl.DataFrame:
    """Return a canonical Polars frame for calibration-like episode rows."""
    return pl.DataFrame(
        [_canonical_episode_row(row) for row in rows],
        schema=EPISODE_SCHEMA,
    )


def episode_rows_lazy(rows: Iterable[Mapping[str, Any]]) -> pl.LazyFrame:
    """Return a lazy canonical episode table for larger artifact pipelines."""
    return episode_rows_frame(rows).lazy()


def summarize_episode_rows(
    rows: Iterable[Mapping[str, Any]] | pl.DataFrame | pl.LazyFrame,
) -> dict[str, Any]:
    """Summarize placement distribution, survivor rate, and final-round stats."""
    frame = _collect_frame(rows)
    if frame.is_empty():
        return {
            "episodes": 0,
            "placement_counts": {},
            "survivor_rate": 0.0,
            "mean_final_round": 0.0,
            "median_final_round": 0.0,
            "min_final_round": 0,
            "max_final_round": 0,
        }

    placement_counts = {
        str(row["placement"]): int(row["episodes"])
        for row in frame.group_by("placement")
        .agg(pl.len().alias("episodes"))
        .sort("placement")
        .iter_rows(named=True)
    }
    round_stats = frame.select(
        pl.len().alias("episodes"),
        pl.col("final_round").mean().alias("mean_final_round"),
        pl.col("final_round").median().alias("median_final_round"),
        pl.col("final_round").min().alias("min_final_round"),
        pl.col("final_round").max().alias("max_final_round"),
        (pl.col("final_reason") == "max_round").mean().alias("survivor_rate"),
    ).row(0, named=True)

    return {
        "episodes": int(round_stats["episodes"]),
        "placement_counts": placement_counts,
        "survivor_rate": float(round_stats["survivor_rate"] or 0.0),
        "mean_final_round": float(round_stats["mean_final_round"] or 0.0),
        "median_final_round": float(round_stats["median_final_round"] or 0.0),
        "min_final_round": int(round_stats["min_final_round"] or 0),
        "max_final_round": int(round_stats["max_final_round"] or 0),
    }


def _collect_frame(rows: Iterable[Mapping[str, Any]] | pl.DataFrame | pl.LazyFrame) -> pl.DataFrame:
    if isinstance(rows, pl.LazyFrame):
        return rows.collect()
    if isinstance(rows, pl.DataFrame):
        return rows
    return episode_rows_frame(rows)


def _canonical_episode_row(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "episode": row.get("episode"),
        "policy": row.get("policy"),
        "placement": row.get("placement", row.get("placement_proxy")),
        "final_round": row.get("final_round", row.get("round")),
        "hp": row.get("hp"),
        "final_reason": row.get("final_reason"),
        "scenario_score": row.get("scenario_score"),
        "illegal_actions": row.get("illegal_actions", row.get("total_illegal_actions")),
    }

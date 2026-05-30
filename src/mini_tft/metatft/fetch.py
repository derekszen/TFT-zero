"""Fetch current-patch MetaTFT catalog payloads from JSON endpoints."""

from __future__ import annotations

from pathlib import Path
from urllib.parse import urlencode

from mini_tft.fight_model.metatft_data import (
    DEFAULT_RANKS,
    METATFT_COMPS_API,
    _get_json,
    fetch_current_comp_strength,
    write_comp_strength_snapshot,
)

METATFT_STAT_API = "https://api-hc.metatft.com/tft-stat-api"
METATFT_LOOKUPS = "https://data.metatft.com/lookups"


def fetch_current_rich_catalog(
    *,
    queue: str = "1100",
    days: int = 3,
    ranks: tuple[str, ...] = DEFAULT_RANKS,
    min_count: int = 3_000,
    comp_detail_limit: int = 12,
) -> dict[str, object]:
    """Fetch comp rankings plus item/build/trait/line endpoint payloads.

    The public site currently exposes JSON endpoints for these fields, so this
    path intentionally avoids HTML scraping and does not require Scrapling.
    """

    payload = fetch_current_comp_strength(
        queue=queue,
        days=days,
        ranks=ranks,
        min_count=min_count,
    )
    source = payload["source"]
    if not isinstance(source, dict):
        raise TypeError("source must be a dict")
    cluster_id = str(source["cluster_id"])
    tft_set = str(source["tft_set"])
    rich: dict[str, object] = {
        "comp_builds": _get_json(_url("comp_builds", cluster_id=cluster_id, queue=queue)),
        "comp_options": _get_json(_url("comp_options", cluster_id=cluster_id, queue=queue)),
        "comp_augments": _get_json(_url("comp_augments", cluster_id=cluster_id, queue=queue)),
        "unit_items_processed": _get_json(f"{METATFT_COMPS_API}/unit_items_processed"),
        "stat_items": _get_json(f"{METATFT_STAT_API}/items"),
        "tables": _get_json(f"{METATFT_LOOKUPS}/latest_{tft_set}_tables.json"),
    }
    records = payload["records"]
    if not isinstance(records, list):
        raise TypeError("records must be a list")
    rich["comp_details"] = {
        str(record["cluster_id"]): _get_json(
            _url(
                "comp_details",
                comp=str(record["cluster_id"]),
                cluster_id=cluster_id,
                queue=queue,
            )
        )
        for record in records[:comp_detail_limit]
        if isinstance(record, dict) and record.get("cluster_id")
    }
    payload["rich"] = rich
    return payload


def write_rich_catalog_snapshot(path: Path, payload: dict[str, object]) -> None:
    write_comp_strength_snapshot(path, payload)


def _url(endpoint: str, **params: str) -> str:
    filtered = {key: value for key, value in params.items() if value}
    return f"{METATFT_COMPS_API}/{endpoint}?{urlencode(filtered)}"

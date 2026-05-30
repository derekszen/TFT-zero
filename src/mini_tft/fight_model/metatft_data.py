"""MetaTFT aggregate comp-line data loading and snapshot helpers."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from mini_tft.fight_model.calibration import CompStrengthRecord

METATFT_COMPS_API = "https://api-hc.metatft.com/tft-comps-api"
DEFAULT_RANKS = (
    "CHALLENGER",
    "DIAMOND",
    "EMERALD",
    "GRANDMASTER",
    "MASTER",
    "PLATINUM",
)


def fetch_current_comp_strength(
    *,
    queue: str = "1100",
    days: int = 3,
    ranks: tuple[str, ...] = DEFAULT_RANKS,
    min_count: int = 10_000,
) -> dict[str, object]:
    """Fetch and normalize current-patch MetaTFT comp aggregate rankings."""

    stats_params = {
        "queue": queue,
        "patch": "current",
        "days": str(days),
        "rank": ",".join(ranks),
        "permit_filter_adjustment": "true",
    }
    stats_url = f"{METATFT_COMPS_API}/comps_stats?{urlencode(stats_params)}"
    cluster_url = f"{METATFT_COMPS_API}/latest_cluster_info"
    stats = _get_json(stats_url)
    cluster_info = _get_json(cluster_url)["cluster_info"]
    clusters = {
        str(row["Cluster"]): row
        for row in cluster_info["cluster_details"]["clusters"]
    }
    records = []
    for row in stats["results"]:
        cluster_id = str(row.get("cluster", ""))
        if not cluster_id or cluster_id not in clusters:
            continue
        places = [int(count) for count in row["places"][:8]]
        count = sum(places)
        if count < min_count:
            continue
        cluster = clusters[cluster_id]
        avg_placement = sum((index + 1) * count for index, count in enumerate(places)) / count
        records.append(
            {
                "cluster_id": cluster_id,
                "name": str(cluster.get("name_string", "")),
                "units": _split_csv(cluster.get("units_string", "")),
                "avg_placement": round(avg_placement, 4),
                "top4_rate": round(sum(places[:4]) / count, 4),
                "win_rate": round(places[0] / count, 4),
                "count": count,
            }
        )
    records.sort(key=lambda record: float(record["avg_placement"]))
    return {
        "source": {
            "name": "MetaTFT current-patch comp stats",
            "retrieved_at": datetime.now(UTC).date().isoformat(),
            "queue": queue,
            "patch": "current",
            "days": days,
            "ranks": list(ranks),
            "stats_url": stats_url,
            "cluster_url": cluster_url,
            "cluster_id": cluster_info["cluster_id"],
            "tft_set": cluster_info["tft_set"],
            "min_count": min_count,
            "updated": stats.get("updated"),
        },
        "records": records,
    }


def write_comp_strength_snapshot(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def records_from_payload(payload: dict[str, object]) -> list[CompStrengthRecord]:
    rows = payload["records"]
    if not isinstance(rows, list):
        raise TypeError("payload records must be a list")
    records = []
    for row in rows:
        if not isinstance(row, dict):
            raise TypeError("record rows must be objects")
        records.append(
            CompStrengthRecord(
                cluster_id=str(row["cluster_id"]),
                name=str(row["name"]),
                units=tuple(str(unit) for unit in row["units"]),
                avg_placement=float(row["avg_placement"]),
                top4_rate=float(row["top4_rate"]),
                win_rate=float(row["win_rate"]),
                count=int(row["count"]),
            )
        )
    return records


def _get_json(url: str) -> dict[str, object]:
    request = Request(url, headers={"User-Agent": "Mozilla/5.0", "Accept": "application/json"})
    with urlopen(request, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def _split_csv(value: object) -> list[str]:
    if not value:
        return []
    return [part.strip() for part in str(value).split(",") if part.strip()]

"""Recurring MiniTFT browser/session regression gate."""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mini_tft.core.actions import Action
from mini_tft.core.state import UnitInstance
from mini_tft.web.server import MiniTFTWebSession


@dataclass(frozen=True)
class WebUiGateConfig:
    seed: int = 0


def run_web_ui_regression_gate(config: WebUiGateConfig) -> dict[str, Any]:
    """Run fast session-level checks for the interactive browser UI payload."""

    started = time.perf_counter()
    checks = [
        _opening_state_check(config.seed),
        _enemy_preview_schedule_check(config.seed),
        _item_action_flow_check(config.seed),
        _manual_move_check(config.seed),
    ]
    failures = [check["name"] for check in checks if check["status"] != "pass"]
    return {
        "status": "pass" if not failures else "fail",
        "seed": config.seed,
        "elapsed_sec": round(time.perf_counter() - started, 3),
        "checks": checks,
        "failures": failures,
    }


def format_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# MiniTFT Web UI Regression Gate",
        "",
        f"**Status:** `{report['status']}`",
        f"**Seed:** `{report['seed']}`",
        f"**Elapsed:** `{report['elapsed_sec']:.3f}s`",
        "",
        "## Checks",
        "",
        "| Check | Status | Key Metrics |",
        "| --- | --- | --- |",
    ]
    for check in report["checks"]:
        lines.append(
            f"| `{check['name']}` | `{check['status']}` | "
            f"{_format_metrics(check['metrics'])} |"
        )
    if report["failures"]:
        lines.extend(["", "## Failures", ""])
        lines.extend(f"- `{failure}`" for failure in report["failures"])
    lines.extend(
        [
            "",
            "## Gate Meaning",
            "",
            "- `opening_state` catches bad HP/gold/stage/shop/board defaults.",
            "- `enemy_preview_schedule` catches enemy boards collapsing to two visible units.",
            "- `item_action_flow` catches unclear or broken component combine/slam UI states.",
            "- `manual_move` catches drag/drop API regressions for bench and board movement.",
        ]
    )
    return "\n".join(lines) + "\n"


def _opening_state_check(seed: int) -> dict[str, Any]:
    session = MiniTFTWebSession(seed=seed)
    payload = session.payload()
    ok = (
        payload["status"]["stage_label"] == "Stage 1-1"
        and payload["status"]["round_type"] == "pve"
        and payload["status"]["hp"] == 100
        and payload["status"]["gold"] == 3
        and payload["enemy"]["unit_count"] == 3
        and len(payload["shop"]) == 5
        and len(payload["board"]) == 9
        and len(payload["bench"]) == 9
        and payload["item_action"]["mode"] == "none"
        and payload["actions"][Action.END_TURN]["legal"]
    )
    return _check(
        "opening_state",
        ok,
        {
            "stage": payload["status"]["stage_label"],
            "gold": payload["status"]["gold"],
            "enemy_units": payload["enemy"]["unit_count"],
            "item_action": payload["item_action"]["label"],
        },
    )


def _enemy_preview_schedule_check(seed: int) -> dict[str, Any]:
    session = MiniTFTWebSession(seed=seed)
    assert session.env.state is not None
    expected = {
        1: ("Stage 1-1", "pve", 3, None, "Minion 1"),
        4: ("Stage 2-1", "pvp", 3, 3, "Enemy 1"),
        7: ("Stage 2-4", "pvp", 4, 4, "Enemy 1"),
        18: ("Stage 4-1", "pvp", 7, 7, "Enemy 1"),
        24: ("Stage 4-7", "pve", 5, None, "Raptor 1"),
        32: ("Stage 6-1", "pvp", 9, 9, "Enemy 1"),
    }
    rows: list[dict[str, Any]] = []
    ok = True
    for round_num, expected_row in expected.items():
        session.env.state.round = round_num
        payload = session.payload()
        actual = (
            payload["status"]["stage_label"],
            payload["status"]["round_type"],
            payload["enemy"]["unit_count"],
            payload["enemy"]["display_level"],
            payload["enemy"]["slots"][0]["name"],
        )
        rows.append({"round": round_num, "actual": actual, "expected": expected_row})
        ok = ok and actual == expected_row and len(payload["enemy"]["slots"]) == expected_row[2]
    return _check(
        "enemy_preview_schedule",
        ok,
        {
            "checked_rounds": len(rows),
            "max_enemy_units": max(row["actual"][2] for row in rows),
            "rows": rows,
        },
    )


def _item_action_flow_check(seed: int) -> dict[str, Any]:
    session = MiniTFTWebSession(seed=seed)
    assert session.env.state is not None
    session.env.state.board[0] = UnitInstance(unit_id=2)
    session.env.state.item_bench = [101, 102]

    component_payload = session.payload()
    completed_payload = session.step(Action.SLAM_BEST_ITEM)
    slammed_payload = session.step(Action.SLAM_BEST_ITEM)

    ok = (
        component_payload["item_action"]["mode"] == "combine"
        and component_payload["item_action"]["label"] == "Combine Rageblade"
        and component_payload["item_action"]["legal"]
        and completed_payload["item_action"]["mode"] == "slam"
        and completed_payload["item_action"]["label"] == "Slam Rageblade"
        and completed_payload["item_action"]["target_name"] == "Vayne"
        and slammed_payload["items"] == []
        and slammed_payload["board"][0]["items"][0]["name"] == "Rageblade"
        and slammed_payload["item_action"]["mode"] == "none"
    )
    return _check(
        "item_action_flow",
        ok,
        {
            "first_label": component_payload["item_action"]["label"],
            "second_label": completed_payload["item_action"]["label"],
            "target": completed_payload["item_action"].get("target_name"),
            "equipped": slammed_payload["board"][0]["items"][0]["name"],
        },
    )


def _manual_move_check(seed: int) -> dict[str, Any]:
    session = MiniTFTWebSession(seed=seed)
    assert session.env.state is not None
    session.env.state.level = 1
    session.env.state.board[0] = UnitInstance(unit_id=1)
    session.env.state.bench[0] = UnitInstance(unit_id=2)

    failed = session.move_unit("bench", 0, "board", 1)
    swapped = session.move_unit("bench", 0, "board", 0)

    ok = (
        failed["last"]["legal"] is False
        and failed["bench"][0]["id"] == 2
        and swapped["last"]["legal"] is True
        and swapped["board"][0]["id"] == 2
        and swapped["bench"][0]["id"] == 1
    )
    return _check(
        "manual_move",
        ok,
        {
            "full_board_move_legal": failed["last"]["legal"],
            "swap_legal": swapped["last"]["legal"],
            "board_1": swapped["board"][0]["name"],
            "bench_1": swapped["bench"][0]["name"],
        },
    )


def _check(name: str, ok: bool, metrics: dict[str, Any]) -> dict[str, Any]:
    return {"name": name, "status": "pass" if ok else "fail", "metrics": metrics}


def _format_metrics(metrics: dict[str, Any]) -> str:
    compact = []
    for key, value in metrics.items():
        if key == "rows":
            compact.append(f"{key}={len(value)}")
        else:
            compact.append(f"{key}={value}")
    return ", ".join(compact)


def _json_default(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return list(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the MiniTFT web UI regression gate.")
    parser.add_argument("--format", choices=["markdown", "json"], default="markdown")
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(list(argv) if argv is not None else None)

    report = run_web_ui_regression_gate(WebUiGateConfig(seed=args.seed))
    if args.format == "json":
        print(json.dumps(report, indent=2, sort_keys=True, default=_json_default))
    else:
        print(format_markdown(report), end="")
    return 1 if args.strict and report["status"] != "pass" else 0


if __name__ == "__main__":
    raise SystemExit(main())

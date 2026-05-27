"""Small HTTP server for the interactive MiniTFT browser UI."""

from __future__ import annotations

import argparse
import json
import mimetypes
from dataclasses import dataclass, field
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import numpy as np

from mini_tft.bots.greedy_board_bot import GreedyBoardBot
from mini_tft.core.actions import NUM_ACTIONS, Action, action_name
from mini_tft.core.combat import board_strength
from mini_tft.core.config import EnvConfig
from mini_tft.core.economy import xp_needed
from mini_tft.core.env import MiniTFTEnv
from mini_tft.core.ids import EMPTY
from mini_tft.core.state import UnitInstance
from mini_tft.core.traits import active_trait_effects, trait_counts

STATIC_DIR = Path(__file__).resolve().parent / "static"
REPO_ROOT = Path(__file__).resolve().parents[3]
ASSETS_ROOT = REPO_ROOT / "assets"
MANIFEST_PATH = ASSETS_ROOT / "manifest.json"

ACTION_LABELS = {
    Action.END_TURN: "End Turn",
    Action.ROLL: "Refresh",
    Action.BUY_XP: "Buy XP",
    Action.BUY_SHOP_0: "Buy Slot 1",
    Action.BUY_SHOP_1: "Buy Slot 2",
    Action.BUY_SHOP_2: "Buy Slot 3",
    Action.BUY_SHOP_3: "Buy Slot 4",
    Action.BUY_SHOP_4: "Buy Slot 5",
    Action.SELL_BENCH_0: "Sell Bench 1",
    Action.SELL_BENCH_1: "Sell Bench 2",
    Action.SELL_BENCH_2: "Sell Bench 3",
    Action.SELL_BENCH_3: "Sell Bench 4",
    Action.SELL_BENCH_4: "Sell Bench 5",
    Action.SELL_BENCH_5: "Sell Bench 6",
    Action.SELL_BENCH_6: "Sell Bench 7",
    Action.SELL_BENCH_7: "Sell Bench 8",
    Action.SELL_BENCH_8: "Sell Bench 9",
    Action.FIELD_BEST_BOARD: "Field Best",
    Action.SLAM_BEST_ITEM: "Slam Item",
}


@dataclass
class MiniTFTWebSession:
    """Own one simulator instance for the local browser UI."""

    seed: int | None = 0
    env: MiniTFTEnv = field(init=False)
    rng: np.random.Generator = field(init=False)
    bot: GreedyBoardBot = field(default_factory=GreedyBoardBot)
    obs: np.ndarray = field(init=False)
    last_reward: float = 0.0
    last_action: int | None = None
    last_legal: bool | None = None
    log: list[str] = field(default_factory=list)
    asset_manifest: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.asset_manifest = _load_asset_manifest()
        self.reset(self.seed)

    def reset(self, seed: int | None = None) -> dict[str, Any]:
        self.seed = self.seed if seed is None else seed
        self.env = MiniTFTEnv(EnvConfig(seed=self.seed))
        self.rng = np.random.default_rng(self.seed)
        self.obs, _ = self.env.reset(seed=self.seed)
        self.last_reward = 0.0
        self.last_action = None
        self.last_legal = None
        self.log = [f"Reset seed {self.seed}"]
        return self.payload()

    def step(self, action: int) -> dict[str, Any]:
        state = self.env.state
        if state is None:
            return self.reset(self.seed)
        if state.done:
            self._append_log("Episode is done. Reset to continue.")
            return self.payload()

        self.obs, reward, terminated, truncated, info = self.env.step(action)
        self.last_reward = float(reward)
        self.last_action = int(action)
        self.last_legal = bool(info.get("legal_action", False))
        result = "legal" if self.last_legal else "illegal"
        suffix = ""
        if terminated or truncated:
            suffix = f" -> done: {self.env.state.final_reason if self.env.state else 'unknown'}"
        action_label = ACTION_LABELS.get(Action(action), action_name(action))
        self._append_log(f"{action_label}: {result}, reward {reward:.3f}{suffix}")
        return self.payload()

    def bot_step(self) -> dict[str, Any]:
        if self.env.state is None or self.env.state.done:
            return self.payload()
        action = self.bot.act(self.env, self.obs, self.rng)
        return self.step(action)

    def payload(self) -> dict[str, Any]:
        return serialize_state(
            self.env,
            asset_manifest=self.asset_manifest,
            seed=self.seed,
            last_reward=self.last_reward,
            last_action=self.last_action,
            last_legal=self.last_legal,
            log=self.log,
        )

    def _append_log(self, message: str) -> None:
        self.log.insert(0, message)
        del self.log[20:]


def serialize_state(
    env: MiniTFTEnv,
    *,
    asset_manifest: dict[str, Any],
    seed: int | None,
    last_reward: float,
    last_action: int | None,
    last_legal: bool | None,
    log: list[str],
) -> dict[str, Any]:
    state = env._require_state()
    stats = board_strength(state.board, env.data)
    enemy_index = min(state.round - 1, len(env.data.enemy_curve) - 1)
    stage = _stage_info(state.round)
    enemy_next = round(env.data.enemy_curve[enemy_index], 2)
    mask = env.action_masks()
    active = active_trait_effects(state.board, env.data)
    counts = trait_counts(state.board, env.data)

    return {
        "seed": seed,
        "status": {
            "round": state.round,
            "stage": stage["stage"],
            "stage_round": stage["stage_round"],
            "stage_label": stage["label"],
            "max_round": env.config.max_round,
            "hp": state.hp,
            "gold": state.gold,
            "level": state.level,
            "xp": state.xp,
            "xp_needed": xp_needed(state.level),
            "done": state.done,
            "final_reason": state.final_reason,
            "step_count": state.step_count,
            "last_win": state.last_win,
            "last_board_strength": round(state.last_board_strength, 2),
            "last_enemy_strength": round(state.last_enemy_strength, 2),
            "strength": round(stats.strength, 2),
            "enemy_next": enemy_next,
            "enemy_power_penalty": round(stats.enemy_power_penalty, 2),
        },
        "enemy": _serialize_enemy(stage["label"], enemy_next),
        "shop": [
            None if unit_id == EMPTY else _serialize_unit_def(env, unit_id, asset_manifest)
            for unit_id in state.shop
        ],
        "bench": [
            _serialize_unit_instance(env, unit, asset_manifest, index)
            for index, unit in enumerate(state.bench)
        ],
        "board": [
            _serialize_unit_instance(env, unit, asset_manifest, index)
            for index, unit in enumerate(state.board)
        ],
        "items": [_serialize_item(env, item_id) for item_id in state.item_bench],
        "traits": _serialize_traits(env, counts, active),
        "actions": [
            {
                "id": index,
                "name": action_name(index),
                "label": ACTION_LABELS.get(Action(index), action_name(index)),
                "legal": bool(mask[index]),
            }
            for index in range(NUM_ACTIONS)
        ],
        "last": {
            "reward": round(last_reward, 4),
            "action": last_action,
            "action_label": None
            if last_action is None
            else ACTION_LABELS.get(Action(last_action), action_name(last_action)),
            "legal": last_legal,
        },
        "log": log,
        "summary": env.episode_summary() if state.done else None,
    }


def _serialize_unit_instance(
    env: MiniTFTEnv,
    unit: UnitInstance | None,
    asset_manifest: dict[str, Any],
    index: int,
) -> dict[str, Any] | None:
    if unit is None:
        return None
    payload = _serialize_unit_def(env, unit.unit_id, asset_manifest)
    payload["slot"] = index
    payload["stars"] = unit.stars
    payload["items"] = [_serialize_item(env, item_id) for item_id in unit.items]
    return payload


def _serialize_unit_def(
    env: MiniTFTEnv,
    unit_id: int,
    asset_manifest: dict[str, Any],
) -> dict[str, Any]:
    unit = env.data.units[unit_id]
    manifest_entry = asset_manifest.get("units", {}).get(str(unit_id), {})
    path = manifest_entry.get("path")
    return {
        "id": unit.id,
        "name": unit.name,
        "cost": unit.cost,
        "role": unit.role,
        "traits": list(unit.traits),
        "base_power": unit.base_power,
        "image": f"/{path}" if path else None,
    }


def _serialize_item(env: MiniTFTEnv, item_id: int) -> dict[str, Any]:
    item = env.data.items[item_id]
    return {
        "id": item.id,
        "name": item.name,
        "tags": list(item.tags),
        "effects": item.effects,
    }


def _serialize_traits(
    env: MiniTFTEnv,
    counts: dict[str, int],
    active: dict[str, dict[str, float]],
) -> list[dict[str, Any]]:
    traits: list[dict[str, Any]] = []
    for trait_id in env.data.trait_ids:
        count = counts[trait_id]
        trait = env.data.traits[trait_id]
        breakpoints = sorted(trait.breakpoints)
        reached = [breakpoint for breakpoint in breakpoints if breakpoint <= count]
        next_breakpoints = [breakpoint for breakpoint in breakpoints if breakpoint > count]
        if count == 0 and trait_id not in active:
            continue
        traits.append(
            {
                "id": trait_id,
                "label": trait_id.replace("_", " ").title(),
                "count": count,
                "breakpoints": breakpoints,
                "active_breakpoint": max(reached) if reached else None,
                "next_breakpoint": min(next_breakpoints) if next_breakpoints else None,
                "active": trait_id in active,
                "effects": active.get(trait_id, {}),
            }
        )
    return sorted(traits, key=lambda item: (not item["active"], -item["count"], item["label"]))


def _stage_info(round_num: int) -> dict[str, int | str]:
    stage = ((round_num - 1) // 6) + 1
    stage_round = ((round_num - 1) % 6) + 1
    return {
        "stage": stage,
        "stage_round": stage_round,
        "label": f"Stage {stage}-{stage_round}",
    }


def _serialize_enemy(stage_label: str, strength: float) -> dict[str, Any]:
    slot_count = min(6, max(2, int(strength // 45) + 2))
    tier = min(5, max(1, int(strength // 70) + 1))
    return {
        "label": f"{stage_label} enemy",
        "strength": strength,
        "slots": [
            {
                "name": f"Enemy {index + 1}",
                "tier": tier,
            }
            for index in range(slot_count)
        ],
    }


def _load_asset_manifest() -> dict[str, Any]:
    if not MANIFEST_PATH.exists():
        return {"units": {}, "items": {}}
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


class MiniTFTHTTPServer(ThreadingHTTPServer):
    def __init__(self, server_address: tuple[str, int], session: MiniTFTWebSession) -> None:
        super().__init__(server_address, MiniTFTRequestHandler)
        self.session = session


class MiniTFTRequestHandler(BaseHTTPRequestHandler):
    server: MiniTFTHTTPServer

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self._serve_file(STATIC_DIR / "index.html")
            return
        if parsed.path.startswith("/static/"):
            self._serve_file(STATIC_DIR / parsed.path.removeprefix("/static/"))
            return
        if parsed.path.startswith("/assets/"):
            self._serve_asset(parsed.path.removeprefix("/assets/"))
            return
        if parsed.path == "/api/state":
            self._send_json(self.server.session.payload())
            return
        self.send_error(HTTPStatus.NOT_FOUND)

    def do_HEAD(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self._serve_file(STATIC_DIR / "index.html", head_only=True)
            return
        if parsed.path.startswith("/static/"):
            self._serve_file(STATIC_DIR / parsed.path.removeprefix("/static/"), head_only=True)
            return
        if parsed.path.startswith("/assets/"):
            self._serve_asset(parsed.path.removeprefix("/assets/"), head_only=True)
            return
        self.send_error(HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/api/reset":
            body = self._read_json()
            query = parse_qs(parsed.query)
            seed = body.get("seed") if isinstance(body, dict) else None
            if seed is None and "seed" in query:
                seed = query["seed"][0]
            self._send_json(self.server.session.reset(_coerce_seed(seed)))
            return
        if parsed.path == "/api/action":
            body = self._read_json()
            action = int(body.get("action", Action.END_TURN)) if isinstance(body, dict) else 0
            self._send_json(self.server.session.step(action))
            return
        if parsed.path == "/api/bot-step":
            self._send_json(self.server.session.bot_step())
            return
        self.send_error(HTTPStatus.NOT_FOUND)

    def log_message(self, format: str, *args: object) -> None:
        del format, args

    def _read_json(self) -> Any:
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0:
            return {}
        raw = self.rfile.read(length).decode("utf-8")
        return json.loads(raw) if raw else {}

    def _serve_file(self, path: Path, *, head_only: bool = False) -> None:
        resolved = path.resolve()
        if not resolved.is_relative_to(STATIC_DIR.resolve()) or not resolved.exists():
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        content = resolved.read_bytes()
        content_type = mimetypes.guess_type(resolved)[0] or "application/octet-stream"
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        if not head_only:
            self.wfile.write(content)

    def _serve_asset(self, relative_path: str, *, head_only: bool = False) -> None:
        resolved = (ASSETS_ROOT / relative_path).resolve()
        if not resolved.is_relative_to(ASSETS_ROOT.resolve()) or not resolved.exists():
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        content = resolved.read_bytes()
        content_type = mimetypes.guess_type(resolved)[0] or "application/octet-stream"
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Cache-Control", "public, max-age=3600")
        self.end_headers()
        if not head_only:
            self.wfile.write(content)

    def _send_json(self, payload: Any, status: HTTPStatus = HTTPStatus.OK) -> None:
        encoded = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(encoded)


def _coerce_seed(value: Any) -> int | None:
    if value is None or value == "":
        return None
    return int(value)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Launch the MiniTFT browser UI.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--seed", type=int, default=0)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    session = MiniTFTWebSession(seed=args.seed)
    server = MiniTFTHTTPServer((args.host, args.port), session)
    url = f"http://{args.host}:{server.server_port}/"
    print(f"MiniTFT browser UI running at {url}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping MiniTFT browser UI.", flush=True)
    finally:
        server.server_close()


if __name__ == "__main__":
    main()

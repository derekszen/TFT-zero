"""Small HTTP server for the interactive MiniTFT browser UI."""

from __future__ import annotations

import argparse
import json
import mimetypes
from dataclasses import dataclass, field
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, cast
from urllib.parse import parse_qs, urlparse

import numpy as np

from mini_tft.bots.greedy_board_bot import GreedyBoardBot
from mini_tft.core.actions import NUM_ACTIONS, Action, action_name
from mini_tft.core.combat import board_strength, enemy_strength_for_round
from mini_tft.core.config import EnvConfig
from mini_tft.core.economy import xp_needed
from mini_tft.core.env import MiniTFTEnv
from mini_tft.core.ids import EMPTY
from mini_tft.core.items import best_item_target, first_combinable_recipe, is_completed_item
from mini_tft.core.rounds import RoundInfo, round_info
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
        action_label = _action_label(action)
        self._append_log(f"{action_label}: {result}, reward {reward:.3f}{suffix}")
        return self.payload()

    def bot_step(self) -> dict[str, Any]:
        if self.env.state is None or self.env.state.done:
            return self.payload()
        action = self.bot.act(self.env, self.obs, self.rng)
        return self.step(action)

    def move_unit(
        self,
        from_zone: str,
        from_index: int,
        to_zone: str,
        to_index: int,
    ) -> dict[str, Any]:
        state = self.env.state
        if state is None:
            return self.reset(self.seed)
        if state.done:
            self._append_log("Episode is done. Reset to move units.")
            return self.payload()

        moved, message = _move_unit_between_slots(
            state.board,
            state.bench,
            max_board_units=state.level,
            from_zone=from_zone,
            from_index=from_index,
            to_zone=to_zone,
            to_index=to_index,
        )
        self.last_reward = 0.0
        self.last_action = None
        self.last_legal = moved
        self._append_log(message)
        if moved:
            self.obs = self.env._observe()
        return self.payload()

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
    stage = round_info(state.round)
    enemy_next = round(enemy_strength_for_round(state.round, env.data, env.config), 2)
    mask = env.action_masks()
    active = active_trait_effects(state.board, env.data)
    counts = trait_counts(state.board, env.data)

    return {
        "seed": seed,
        "status": {
            "round": state.round,
            "stage": stage.stage,
            "stage_round": stage.stage_round,
            "stage_label": stage.stage_label,
            "round_type": stage.round_type,
            "is_pve_round": stage.is_pve,
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
        "enemy": _serialize_enemy(stage, enemy_next),
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
        "item_action": _serialize_item_action(env, mask),
        "traits": _serialize_traits(env, counts, active),
        "actions": [
            {
                "id": index,
                "name": action_name(index),
                "label": _action_label(index),
                "legal": bool(mask[index]),
            }
            for index in range(NUM_ACTIONS)
        ],
        "last": {
            "reward": round(last_reward, 4),
            "action": last_action,
            "action_label": None
            if last_action is None
            else _action_label(last_action),
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
        "kind": item.kind,
        "components": list(item.components),
        "tags": list(item.tags),
        "effects": item.effects,
    }


def _serialize_item_action(env: MiniTFTEnv, mask: np.ndarray) -> dict[str, Any]:
    state = env._require_state()
    legal = bool(mask[Action.SLAM_BEST_ITEM])

    if not state.item_bench:
        return {
            "mode": "none",
            "label": "No Items",
            "detail": "",
            "legal": False,
        }

    completed = [
        (index, item_id)
        for index, item_id in enumerate(state.item_bench)
        if is_completed_item(item_id, env.data)
    ]
    recipe = first_combinable_recipe(state.item_bench, env.data)

    if completed:
        _, item_id = completed[0]
        item = env.data.items[item_id]
        target = best_item_target(state.board, item_id, env.data, env.config)
        if target is not None and state.board[target] is not None:
            target_unit = state.board[target]
            if target_unit is None:
                raise RuntimeError("best item target disappeared while serializing item action")
            unit = env.data.units[target_unit.unit_id]
            return {
                "mode": "slam",
                "label": f"Slam {item.name}",
                "detail": f"{item.name} -> {unit.name}",
                "legal": legal,
                "item_id": item_id,
                "target_slot": target,
                "target_name": unit.name,
            }
        if recipe is not None:
            return _serialize_recipe_action(env, recipe, legal)
        return {
            "mode": "blocked",
            "label": "Field Unit",
            "detail": f"Field a unit to slam {item.name}",
            "legal": False,
            "item_id": item_id,
        }

    if recipe is not None:
        return _serialize_recipe_action(env, recipe, legal)

    return {
        "mode": "blocked",
        "label": "Need Pair",
        "detail": "Need two compatible components",
        "legal": False,
    }


def _serialize_recipe_action(
    env: MiniTFTEnv,
    recipe: tuple[int, int, int],
    legal: bool,
) -> dict[str, Any]:
    first_index, second_index, completed_id = recipe
    state = env._require_state()
    first_item = env.data.items[state.item_bench[first_index]]
    second_item = env.data.items[state.item_bench[second_index]]
    completed_item = env.data.items[completed_id]
    return {
        "mode": "combine",
        "label": f"Combine {completed_item.name}",
        "detail": f"{first_item.name} + {second_item.name} -> {completed_item.name}",
        "legal": legal,
        "item_id": completed_id,
        "component_indexes": [first_index, second_index],
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


def _action_label(action: int) -> str:
    try:
        enum_action = Action(action)
    except ValueError:
        return action_name(action).replace("_", " ").title()
    return ACTION_LABELS.get(enum_action, action_name(action))


def _move_unit_between_slots(
    board: list[UnitInstance | None],
    bench: list[UnitInstance | None],
    *,
    max_board_units: int,
    from_zone: str,
    from_index: int,
    to_zone: str,
    to_index: int,
) -> tuple[bool, str]:
    zones = {"board": board, "bench": bench}
    if from_zone not in zones or to_zone not in zones:
        return False, "Move failed: unknown zone."
    if not _valid_slot(zones[from_zone], from_index) or not _valid_slot(zones[to_zone], to_index):
        return False, "Move failed: slot out of range."
    if from_zone == to_zone and from_index == to_index:
        return False, "Move skipped: same slot."

    source = zones[from_zone]
    target = zones[to_zone]
    unit = source[from_index]
    if unit is None:
        return False, "Move failed: source slot is empty."

    target_unit = target[to_index]
    if to_zone == "board" and from_zone != "board" and target_unit is None:
        board_count = sum(slot is not None for slot in board)
        if board_count >= max_board_units:
            return False, f"Move failed: board is full at level {max_board_units}."

    source[from_index], target[to_index] = target_unit, unit
    unit_name = "unit"
    from_label = _slot_label(from_zone, from_index)
    to_label = _slot_label(to_zone, to_index)
    return True, f"Moved {unit_name}: {from_label} -> {to_label}"


def _valid_slot(slots: list[UnitInstance | None], index: int) -> bool:
    return 0 <= index < len(slots)


def _slot_label(zone: str, index: int) -> str:
    return f"{zone.title()} {index + 1}"


def _serialize_enemy(stage: RoundInfo, strength: float) -> dict[str, Any]:
    slot_count = _enemy_slot_count(stage)
    tier = min(5, max(1, stage.stage - 1 if not stage.is_pve else stage.stage))
    return {
        "label": f"{stage.stage_label} enemy",
        "strength": strength,
        "unit_count": slot_count,
        "display_level": slot_count if not stage.is_pve else None,
        "slots": [
            {
                "name": _enemy_slot_name(stage, index),
                "tier": tier,
            }
            for index in range(slot_count)
        ],
    }


def _enemy_slot_count(stage: RoundInfo) -> int:
    if stage.is_pve:
        return {
            1: 3,
            2: 3,
            3: 4,
            4: 5,
            5: 6,
            6: 1,
        }.get(stage.stage, 6)
    if stage.stage == 2:
        return 3 if stage.stage_round <= 2 else 4
    if stage.stage == 3:
        return 5 if stage.stage_round <= 3 else 6
    if stage.stage == 4:
        return 7
    if stage.stage == 5:
        return 8 if stage.stage_round <= 4 else 9
    return 9


def _enemy_slot_name(stage: RoundInfo, index: int) -> str:
    if not stage.is_pve:
        return f"Enemy {index + 1}"
    if stage.stage == 1:
        return f"Minion {index + 1}"
    if stage.stage == 2:
        return f"Krug {index + 1}"
    if stage.stage == 3:
        return f"Wolf {index + 1}"
    if stage.stage == 4:
        return f"Raptor {index + 1}"
    if stage.stage == 6 and index == 0:
        return "Dragon"
    return f"Monster {index + 1}"


def _load_asset_manifest() -> dict[str, Any]:
    if not MANIFEST_PATH.exists():
        return {"units": {}, "items": {}}
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


class MiniTFTHTTPServer(ThreadingHTTPServer):
    def __init__(self, server_address: tuple[str, int], session: MiniTFTWebSession) -> None:
        super().__init__(server_address, MiniTFTRequestHandler)
        self.session = session


class MiniTFTRequestHandler(BaseHTTPRequestHandler):
    @property
    def _session(self) -> MiniTFTWebSession:
        return cast(MiniTFTHTTPServer, self.server).session

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
            self._send_json(self._session.payload())
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
            self._send_json(self._session.reset(_coerce_seed(seed)))
            return
        if parsed.path == "/api/action":
            body = self._read_json()
            action = int(body.get("action", Action.END_TURN)) if isinstance(body, dict) else 0
            self._send_json(self._session.step(action))
            return
        if parsed.path == "/api/bot-step":
            self._send_json(self._session.bot_step())
            return
        if parsed.path == "/api/move-unit":
            body = self._read_json()
            self._send_json(
                self._session.move_unit(
                    from_zone=str(body.get("from_zone", "")),
                    from_index=int(body.get("from_index", -1)),
                    to_zone=str(body.get("to_zone", "")),
                    to_index=int(body.get("to_index", -1)),
                )
            )
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

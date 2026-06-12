"""Objective schema for MiniTFT automated research runs."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

MetricMode = Literal["higher", "lower"]
PolicyKind = Literal["bot", "checkpoint"]
EnvironmentKind = Literal["single_player", "lobby"]


@dataclass(frozen=True)
class EnvironmentSpec:
    kind: EnvironmentKind
    opponent_policy: str = "fast_level"
    player_count: int = 8
    max_actions_per_player: int | None = None


@dataclass(frozen=True)
class PolicySpec:
    kind: PolicyKind
    name: str
    path: Path | None = None


@dataclass(frozen=True)
class SeedSet:
    name: str
    start: int
    episodes: int

    def seeds(self) -> range:
        return range(self.start, self.start + self.episodes)


@dataclass(frozen=True)
class MetricRule:
    name: str
    mode: MetricMode
    min_delta: float
    required: bool = True


@dataclass(frozen=True)
class SafetySpec:
    protected_paths: tuple[str, ...]
    fail_on_protected_changes: bool = True


@dataclass(frozen=True)
class ObjectiveSpec:
    name: str
    description: str
    environment: EnvironmentSpec
    baseline: PolicySpec
    candidate: PolicySpec
    seed_sets: dict[str, SeedSet]
    metric_rules: tuple[MetricRule, ...]
    safety: SafetySpec
    source_path: Path


def load_objective(path: Path, repo_root: Path) -> ObjectiveSpec:
    """Load an autoresearch objective from TOML."""

    raw: dict[str, Any] = tomllib.loads(path.read_text(encoding="utf-8"))
    baseline = _load_policy(raw["baseline"], repo_root)
    candidate = _load_policy(raw["candidate"], repo_root)
    seed_sets = {
        name: SeedSet(name=name, start=int(spec["start"]), episodes=int(spec["episodes"]))
        for name, spec in raw["seed_sets"].items()
    }
    metric_rules = tuple(
        MetricRule(
            name=str(row["name"]),
            mode=_metric_mode(str(row["mode"])),
            min_delta=float(row["min_delta"]),
            required=bool(row.get("required", True)),
        )
        for row in raw["acceptance"]["metrics"]
    )
    safety_raw = raw.get("safety", {})
    safety = SafetySpec(
        protected_paths=tuple(str(path) for path in safety_raw.get("protected_paths", ())),
        fail_on_protected_changes=bool(safety_raw.get("fail_on_protected_changes", True)),
    )
    return ObjectiveSpec(
        name=str(raw["name"]),
        description=str(raw.get("description", "")),
        environment=_load_environment(raw.get("environment", {})),
        baseline=baseline,
        candidate=candidate,
        seed_sets=seed_sets,
        metric_rules=metric_rules,
        safety=safety,
        source_path=path,
    )


def _load_environment(raw: dict[str, Any]) -> EnvironmentSpec:
    return EnvironmentSpec(
        kind=_environment_kind(str(raw.get("kind", "single_player"))),
        opponent_policy=str(raw.get("opponent_policy", "fast_level")),
        player_count=int(raw.get("player_count", 8)),
        max_actions_per_player=_optional_int(raw.get("max_actions_per_player")),
    )


def _load_policy(raw: dict[str, Any], repo_root: Path) -> PolicySpec:
    kind = _policy_kind(str(raw["kind"]))
    raw_path = raw.get("path")
    path = _resolve_path(str(raw_path), repo_root) if raw_path is not None else None
    return PolicySpec(kind=kind, name=str(raw["name"]), path=path)


def _resolve_path(value: str, repo_root: Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (repo_root / path).resolve()


def _optional_int(value: Any) -> int | None:
    return None if value is None else int(value)


def _policy_kind(value: str) -> PolicyKind:
    if value == "bot":
        return "bot"
    if value == "checkpoint":
        return "checkpoint"
    raise ValueError(f"unknown policy kind: {value}")


def _metric_mode(value: str) -> MetricMode:
    if value == "higher":
        return "higher"
    if value == "lower":
        return "lower"
    raise ValueError(f"unknown metric mode: {value}")


def _environment_kind(value: str) -> EnvironmentKind:
    if value == "single_player":
        return "single_player"
    if value == "lobby":
        return "lobby"
    raise ValueError(f"unknown environment kind: {value}")

"""Normalized catalog for current-patch MetaTFT aggregate data."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

UNIT_NAMESPACE = "metatft_current_patch"


@dataclass(frozen=True)
class CatalogMetadata:
    source_name: str
    tft_set: str
    patch: str
    queue: str
    retrieved_at: str
    cluster_id: str
    stats_url: str
    cluster_url: str
    days: int | None = None
    ranks: tuple[str, ...] = ()


@dataclass(frozen=True)
class CatalogUnit:
    unit_id: int
    key: str
    display_name: str
    cost: int | None = None


@dataclass(frozen=True)
class CatalogTag:
    tag_id: int
    key: str
    display_name: str
    kind: str


@dataclass(frozen=True)
class CatalogItem:
    item_id: int
    key: str
    display_name: str
    avg_placement: float | None = None
    count: int = 0


@dataclass(frozen=True)
class CatalogAugment:
    augment_id: int
    key: str
    display_name: str
    tier: str | None = None
    avg_placement: float | None = None
    count: int = 0


@dataclass(frozen=True)
class TraitBreakpoint:
    trait_key: str
    level: int
    avg_placement: float | None = None
    count: int = 0


@dataclass(frozen=True)
class CatalogTrait:
    trait_id: int
    key: str
    display_name: str
    breakpoints: tuple[TraitBreakpoint, ...] = ()


@dataclass(frozen=True)
class UnitItemBuild:
    unit_key: str
    item_keys: tuple[str, ...]
    avg_placement: float | None = None
    count: int = 0
    score: float = 0.0
    place_change: float | None = None


@dataclass(frozen=True)
class CompStageLine:
    line: str
    stage: int
    stage_round: int
    level: int
    unit_keys: tuple[str, ...]
    trait_keys: tuple[str, ...] = ()
    avg_placement: float | None = None
    count: int = 0
    score: float = 0.0
    source: str = "metatft"


@dataclass(frozen=True)
class MetaTFTComp:
    comp_id: str
    rank: int
    name: str
    unit_keys: tuple[str, ...]
    unit_ids: tuple[int, ...]
    tag_keys: tuple[str, ...]
    tag_ids: tuple[int, ...]
    avg_placement: float
    top4_rate: float
    win_rate: float
    count: int
    item_builds: tuple[UnitItemBuild, ...] = ()
    trait_breakpoints: tuple[TraitBreakpoint, ...] = ()
    stage_lines: tuple[CompStageLine, ...] = ()
    augment_keys: tuple[str, ...] = ()

    @property
    def level_hint(self) -> int:
        return max(1, min(10, len(self.unit_keys)))


@dataclass(frozen=True)
class MetaTFTCatalog:
    metadata: CatalogMetadata
    units: tuple[CatalogUnit, ...]
    tags: tuple[CatalogTag, ...]
    comps: tuple[MetaTFTComp, ...]
    items: tuple[CatalogItem, ...] = ()
    augments: tuple[CatalogAugment, ...] = ()
    traits: tuple[CatalogTrait, ...] = ()
    unit_namespace: str = UNIT_NAMESPACE
    _unit_by_key: dict[str, CatalogUnit] = field(init=False, repr=False)
    _tag_by_key: dict[str, CatalogTag] = field(init=False, repr=False)
    _comp_by_id: dict[str, MetaTFTComp] = field(init=False, repr=False)
    _comp_index_by_id: dict[str, int] = field(init=False, repr=False)
    _item_id_by_key: dict[str, int] = field(init=False, repr=False)
    _augment_id_by_key: dict[str, int] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "_unit_by_key", {unit.key: unit for unit in self.units})
        object.__setattr__(self, "_tag_by_key", {tag.key: tag for tag in self.tags})
        object.__setattr__(self, "_comp_by_id", {comp.comp_id: comp for comp in self.comps})
        object.__setattr__(
            self,
            "_comp_index_by_id",
            {comp.comp_id: index + 1 for index, comp in enumerate(self.comps)},
        )
        object.__setattr__(
            self,
            "_item_id_by_key",
            {item.key: item.item_id for item in self.items},
        )
        object.__setattr__(
            self,
            "_augment_id_by_key",
            {augment.key: augment.augment_id for augment in self.augments},
        )

    @property
    def unit_count(self) -> int:
        return len(self.units)

    @property
    def comp_count(self) -> int:
        return len(self.comps)

    @property
    def tag_count(self) -> int:
        return len(self.tags)

    @property
    def item_count(self) -> int:
        return len(self.items)

    @property
    def augment_count(self) -> int:
        return len(self.augments)

    @property
    def trait_count(self) -> int:
        return len(self.traits)

    def unit_id(self, key: str) -> int:
        try:
            return self._unit_by_key[key].unit_id
        except KeyError as exc:
            raise KeyError(f"unknown MetaTFT unit key: {key}") from exc

    def tag_id(self, key: str) -> int:
        try:
            return self._tag_by_key[key].tag_id
        except KeyError as exc:
            raise KeyError(f"unknown MetaTFT tag key: {key}") from exc

    def item_id(self, key: str) -> int:
        try:
            return self._item_id_by_key[key]
        except KeyError as exc:
            raise KeyError(f"unknown MetaTFT item key: {key}") from exc

    def augment_id(self, key: str) -> int:
        try:
            return self._augment_id_by_key[key]
        except KeyError as exc:
            raise KeyError(f"unknown MetaTFT augment key: {key}") from exc

    def comp(self, comp_id: str) -> MetaTFTComp:
        try:
            return self._comp_by_id[comp_id]
        except KeyError as exc:
            raise KeyError(f"unknown MetaTFT comp id: {comp_id}") from exc

    def comp_index(self, comp_id: str | None) -> int:
        if comp_id is None:
            return 0
        try:
            return self._comp_index_by_id[comp_id]
        except KeyError as exc:
            raise KeyError(f"unknown MetaTFT comp id: {comp_id}") from exc


def load_catalog_from_comp_strength(path: Path) -> MetaTFTCatalog:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return load_catalog_from_payload(payload)


def load_catalog_from_payload(payload: dict[str, Any]) -> MetaTFTCatalog:
    source = payload["source"]
    records = payload["records"]
    if not isinstance(source, dict):
        raise TypeError("MetaTFT source must be an object")
    if not isinstance(records, list):
        raise TypeError("MetaTFT records must be a list")
    rich = payload.get("rich")
    if rich is not None and not isinstance(rich, dict):
        raise TypeError("MetaTFT rich payload must be an object")
    return build_catalog_from_comp_records(source, records, rich=rich)


def build_catalog_from_comp_records(
    source: dict[str, Any],
    records: list[dict[str, Any]],
    *,
    rich: dict[str, Any] | None = None,
) -> MetaTFTCatalog:
    details_by_comp = _details_by_comp(rich)
    unit_keys = sorted(
        {
            *{unit for row in records for unit in _row_units(row)},
            *{unit for details in details_by_comp.values() for unit in _detail_unit_keys(details)},
        }
    )
    unit_costs = _unit_costs(rich)
    units = tuple(
        CatalogUnit(
            index + 1,
            unit_key,
            _display_name(unit_key),
            unit_costs.get(unit_key),
        )
        for index, unit_key in enumerate(unit_keys)
    )
    unit_id_by_key = {unit.key: unit.unit_id for unit in units}

    trait_breakpoints_by_comp = {
        comp_id: _trait_breakpoints_from_details(details)
        for comp_id, details in details_by_comp.items()
    }
    trait_keys = sorted(
        {
            *{
                breakpoint.trait_key
                for breakpoints in trait_breakpoints_by_comp.values()
                for breakpoint in breakpoints
            },
            *{
                trait_key
                for details in details_by_comp.values()
                for trait_key in _detail_trait_keys(details)
            },
        }
    )
    tag_keys = sorted(
        {
            *{tag for row in records for tag in _row_tags(row)},
            *trait_keys,
            *{
                augment
                for details in details_by_comp.values()
                for augment in _detail_augment_keys(details)
            },
        }
    )
    tags = tuple(
        CatalogTag(
            tag_id=index + 1,
            key=tag_key,
            display_name=_display_name(tag_key),
            kind=_tag_kind(tag_key, unit_id_by_key),
        )
        for index, tag_key in enumerate(tag_keys)
    )
    tag_id_by_key = {tag.key: tag.tag_id for tag in tags}

    sorted_records = sorted(records, key=lambda row: float(row["avg_placement"]))
    comps = tuple(
        _comp_from_row(
            row,
            rank,
            unit_id_by_key,
            tag_id_by_key,
            details_by_comp.get(str(row["cluster_id"]), {}),
            trait_breakpoints_by_comp.get(str(row["cluster_id"]), ()),
        )
        for rank, row in enumerate(sorted_records, start=1)
    )
    item_keys = sorted(
        {
            *{item for details in details_by_comp.values() for item in _detail_item_keys(details)},
            *_unit_item_keys(rich),
            *_stat_item_keys(rich),
        }
    )
    item_stats = _item_stats(rich)
    items = tuple(
        CatalogItem(
            item_id=index + 1,
            key=item_key,
            display_name=_display_name(item_key),
            avg_placement=item_stats.get(item_key, (None, 0))[0],
            count=item_stats.get(item_key, (None, 0))[1],
        )
        for index, item_key in enumerate(item_keys)
    )
    augment_keys = sorted(
        {
            *{tag.key for tag in tags if tag.kind == "augment"},
            *_table_augment_keys(rich),
        }
    )
    augments = tuple(
        CatalogAugment(
            augment_id=index + 1,
            key=augment_key,
            display_name=_display_name(augment_key),
            tier=_augment_tier(rich, augment_key),
        )
        for index, augment_key in enumerate(augment_keys)
    )
    trait_breakpoints = _merge_trait_breakpoints(trait_breakpoints_by_comp)
    traits = tuple(
        CatalogTrait(
            trait_id=index + 1,
            key=trait_key,
            display_name=_display_name(trait_key),
            breakpoints=tuple(trait_breakpoints.get(trait_key, ())),
        )
        for index, trait_key in enumerate(trait_keys)
    )
    metadata = CatalogMetadata(
        source_name=str(source.get("name", "")),
        tft_set=str(source.get("tft_set", "")),
        patch=str(source.get("patch", "")),
        queue=str(source.get("queue", "")),
        retrieved_at=str(source.get("retrieved_at", "")),
        cluster_id=str(source.get("cluster_id", "")),
        stats_url=str(source.get("stats_url", "")),
        cluster_url=str(source.get("cluster_url", "")),
        days=int(source["days"]) if source.get("days") is not None else None,
        ranks=tuple(str(rank) for rank in source.get("ranks", ())),
    )
    return MetaTFTCatalog(
        metadata=metadata,
        units=units,
        tags=tags,
        comps=comps,
        items=items,
        augments=augments,
        traits=traits,
    )


def _comp_from_row(
    row: dict[str, Any],
    rank: int,
    unit_id_by_key: dict[str, int],
    tag_id_by_key: dict[str, int],
    details: dict[str, Any],
    trait_breakpoints: tuple[TraitBreakpoint, ...],
) -> MetaTFTComp:
    unit_keys = tuple(_row_units(row))
    tag_keys = tuple(_row_tags(row))
    augment_keys = tuple(tag for tag in tag_keys if _tag_kind(tag, unit_id_by_key) == "augment")
    return MetaTFTComp(
        comp_id=str(row["cluster_id"]),
        rank=rank,
        name=str(row["name"]),
        unit_keys=unit_keys,
        unit_ids=tuple(unit_id_by_key[unit_key] for unit_key in unit_keys),
        tag_keys=tag_keys,
        tag_ids=tuple(tag_id_by_key[tag_key] for tag_key in tag_keys),
        avg_placement=float(row["avg_placement"]),
        top4_rate=float(row["top4_rate"]),
        win_rate=float(row["win_rate"]),
        count=int(row["count"]),
        item_builds=_item_builds_from_details(details),
        trait_breakpoints=trait_breakpoints,
        stage_lines=_stage_lines_from_details(str(row["cluster_id"]), details),
        augment_keys=augment_keys,
    )


def _row_units(row: dict[str, Any]) -> list[str]:
    units = row.get("units", ())
    if not isinstance(units, list | tuple):
        raise TypeError("MetaTFT comp units must be a list")
    return [str(unit) for unit in units if unit]


def _row_tags(row: dict[str, Any]) -> list[str]:
    tags = []
    for token in str(row.get("name", "")).split(","):
        stripped = token.strip()
        if stripped:
            tags.append(stripped)
    return tags


def _tag_kind(tag_key: str, unit_id_by_key: dict[str, int]) -> str:
    lower = tag_key.lower()
    if tag_key in unit_id_by_key:
        return "unit"
    if "augment" in lower:
        return "augment"
    if lower.endswith("trait") or "trait" in lower:
        return "trait"
    if lower.endswith("tank") or "tank" in lower:
        return "archetype"
    return "other"


def _details_by_comp(rich: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    if not rich:
        return {}
    normalized: dict[str, dict[str, Any]] = {}
    details = rich.get("comp_details", {})
    if isinstance(details, dict):
        for comp_id, payload in details.items():
            if isinstance(payload, dict):
                results = payload.get("results", payload)
                if isinstance(results, dict):
                    normalized[str(comp_id)] = dict(results)

    comp_builds = rich.get("comp_builds", {})
    build_results = comp_builds.get("results", {}) if isinstance(comp_builds, dict) else {}
    if isinstance(build_results, dict):
        for comp_id, payload in build_results.items():
            if not isinstance(payload, dict):
                continue
            comp_details = normalized.setdefault(str(comp_id), {"cluster": str(comp_id)})
            if "builds" in payload:
                comp_details.setdefault("builds", payload["builds"])

    comp_options = rich.get("comp_options", {})
    option_results = comp_options.get("results", {}) if isinstance(comp_options, dict) else {}
    options = option_results.get("options", {}) if isinstance(option_results, dict) else {}
    if isinstance(options, dict):
        for comp_id, payload in options.items():
            if isinstance(payload, dict):
                normalized.setdefault(str(comp_id), {"cluster": str(comp_id)}).setdefault(
                    "options",
                    payload,
                )

    comp_augments = rich.get("comp_augments", {})
    augment_results = comp_augments.get("results", {}) if isinstance(comp_augments, dict) else {}
    augment_stats = (
        augment_results.get("augment_stats", {}) if isinstance(augment_results, dict) else {}
    )
    if isinstance(augment_stats, dict):
        for comp_id, payload in augment_stats.items():
            if isinstance(payload, list):
                normalized.setdefault(str(comp_id), {"cluster": str(comp_id)}).setdefault(
                    "augments",
                    payload,
                )
    return normalized


def _detail_unit_keys(details: dict[str, Any]) -> set[str]:
    units: set[str] = set()
    for row in details.get("unit_stats", ()):
        if isinstance(row, dict) and row.get("unit"):
            units.add(str(row["unit"]))
    for line in _iter_option_rows(details):
        units.update(_split_ampersand(line.get("units_list") or line.get("unit_list")))
    for build in details.get("builds", ()):
        if isinstance(build, dict) and build.get("unit"):
            units.add(str(build["unit"]))
    return units


def _detail_trait_keys(details: dict[str, Any]) -> set[str]:
    traits: set[str] = set()
    for row in details.get("traits", ()):
        if isinstance(row, dict) and row.get("trait"):
            traits.add(str(row["trait"]))
    for line in _iter_option_rows(details):
        traits.update(
            _trait_key_from_token(token) for token in _split_ampersand(line.get("traits_list"))
        )
    return {trait for trait in traits if trait}


def _detail_augment_keys(details: dict[str, Any]) -> set[str]:
    augments = set()
    for row in details.get("augments", ()):
        if isinstance(row, dict) and row.get("aug"):
            augments.add(str(row["aug"]))
    return augments


def _detail_item_keys(details: dict[str, Any]) -> set[str]:
    item_keys: set[str] = set()
    for build in details.get("builds", ()):
        if isinstance(build, dict):
            item_keys.update(str(item) for item in build.get("buildName", ()) if item)
    for row in details.get("itemNames", ()):
        if isinstance(row, dict):
            item_keys.update(_split_ampersand(row.get("itemNames")))
    return item_keys


def _unit_item_keys(rich: dict[str, Any] | None) -> set[str]:
    if not rich:
        return set()
    payload = rich.get("unit_items_processed", {})
    item_names = payload.get("itemNames", ()) if isinstance(payload, dict) else ()
    if isinstance(item_names, dict):
        return {str(item) for item in item_names}
    if isinstance(item_names, list | tuple):
        return {str(item) for item in item_names}
    return set()


def _unit_costs(rich: dict[str, Any] | None) -> dict[str, int]:
    if not rich:
        return {}
    costs = rich.get("unit_costs", {})
    if not isinstance(costs, dict):
        return {}
    return {
        str(unit_key): max(0, int(cost))
        for unit_key, cost in costs.items()
        if unit_key and cost is not None
    }


def _stat_item_keys(rich: dict[str, Any] | None) -> set[str]:
    if not rich:
        return set()
    payload = rich.get("stat_items", {})
    rows = payload.get("results", ()) if isinstance(payload, dict) else ()
    return {
        str(row["itemName"])
        for row in rows
        if isinstance(row, dict) and row.get("itemName")
    }


def _table_augment_keys(rich: dict[str, Any] | None) -> set[str]:
    if not rich:
        return set()
    tables = rich.get("tables", {})
    effects = tables.get("itemEffects", {}) if isinstance(tables, dict) else {}
    if not isinstance(effects, dict):
        return set()
    return {str(key) for key in effects if "Augment" in str(key)}


def _augment_tier(rich: dict[str, Any] | None, augment_key: str) -> str | None:
    if not rich:
        return None
    tables = rich.get("tables", {})
    effects = tables.get("itemEffects", {}) if isinstance(tables, dict) else {}
    row = effects.get(augment_key) if isinstance(effects, dict) else None
    if isinstance(row, dict):
        tier = row.get("tier") or row.get("rarity")
        return str(tier) if tier is not None else None
    return None


def _item_stats(rich: dict[str, Any] | None) -> dict[str, tuple[float | None, int]]:
    stats: dict[str, tuple[float | None, int]] = {}
    if not rich:
        return stats
    payload = rich.get("stat_items", {})
    rows = payload.get("results", ()) if isinstance(payload, dict) else ()
    for row in rows:
        if not isinstance(row, dict) or not row.get("itemName"):
            continue
        places = [int(value) for value in row.get("places", ())[:8]]
        count = sum(places)
        avg = (
            sum((index + 1) * value for index, value in enumerate(places)) / count
            if count
            else None
        )
        stats[str(row["itemName"])] = (avg, count)
    return stats


def _trait_breakpoints_from_details(details: dict[str, Any]) -> tuple[TraitBreakpoint, ...]:
    breakpoints: list[TraitBreakpoint] = []
    for row in details.get("traits", ()):
        if not isinstance(row, dict) or not row.get("trait"):
            continue
        trait_key = str(row["trait"])
        for level in row.get("levels", ()):
            if not isinstance(level, dict):
                continue
            breakpoints.append(
                TraitBreakpoint(
                    trait_key=trait_key,
                    level=int(level["level"]),
                    avg_placement=float(level["avg"]) if level.get("avg") is not None else None,
                    count=int(level.get("count", 0)),
                )
            )
    for line in _iter_option_rows(details):
        for token in _split_ampersand(line.get("traits_list")):
            trait_key = _trait_key_from_token(token)
            level = _trait_level_from_token(token)
            if trait_key and level is not None:
                breakpoints.append(
                    TraitBreakpoint(
                        trait_key=trait_key,
                        level=level,
                        avg_placement=float(line["avg"]) if line.get("avg") is not None else None,
                        count=int(line.get("count", 0)),
                    )
                )
    dedup: dict[tuple[str, int, float | None], TraitBreakpoint] = {}
    for breakpoint in breakpoints:
        key = (breakpoint.trait_key, breakpoint.level, breakpoint.avg_placement)
        if key not in dedup or breakpoint.count > dedup[key].count:
            dedup[key] = breakpoint
    return tuple(sorted(dedup.values(), key=lambda bp: (bp.trait_key, bp.level, -bp.count)))


def _merge_trait_breakpoints(
    breakpoints_by_comp: dict[str, tuple[TraitBreakpoint, ...]],
) -> dict[str, list[TraitBreakpoint]]:
    merged: dict[tuple[str, int], TraitBreakpoint] = {}
    for breakpoints in breakpoints_by_comp.values():
        for breakpoint in breakpoints:
            key = (breakpoint.trait_key, breakpoint.level)
            existing = merged.get(key)
            if existing is None:
                merged[key] = breakpoint
                continue
            count = existing.count + breakpoint.count
            avg = None
            if (
                existing.avg_placement is not None
                and breakpoint.avg_placement is not None
                and count
            ):
                avg = (
                    existing.avg_placement * existing.count
                    + breakpoint.avg_placement * breakpoint.count
                ) / count
            merged[key] = TraitBreakpoint(breakpoint.trait_key, breakpoint.level, avg, count)
    grouped: dict[str, list[TraitBreakpoint]] = {}
    for breakpoint in merged.values():
        grouped.setdefault(breakpoint.trait_key, []).append(breakpoint)
    for trait_key in grouped:
        grouped[trait_key].sort(key=lambda bp: bp.level)
    return grouped


def _item_builds_from_details(details: dict[str, Any]) -> tuple[UnitItemBuild, ...]:
    builds = []
    for row in details.get("builds", ()):
        if not isinstance(row, dict) or not row.get("unit"):
            continue
        item_keys = tuple(str(item) for item in row.get("buildName", ()) if item)
        if not item_keys:
            continue
        builds.append(
            UnitItemBuild(
                unit_key=str(row["unit"]),
                item_keys=item_keys,
                avg_placement=float(row["avg"]) if row.get("avg") is not None else None,
                count=int(row.get("count", 0)),
                score=float(row.get("score", 0.0)),
                place_change=(
                    float(row["place_change"]) if row.get("place_change") is not None else None
                ),
            )
        )
    builds.sort(key=lambda build: (-build.score, build.avg_placement or 99.0, -build.count))
    return tuple(builds)


def _stage_lines_from_details(comp_id: str, details: dict[str, Any]) -> tuple[CompStageLine, ...]:
    lines: list[CompStageLine] = []
    for level_key, rows in _option_groups(details.get("early_options")).items():
        row = _best_option(rows, prefer="win")
        if row is None:
            continue
        level = int(float(row.get("level", level_key)))
        lines.append(
            CompStageLine(
                line=f"early_{level_key}",
                stage=2 if level <= 5 else 3,
                stage_round=1 if level <= 4 else 2,
                level=max(1, min(10, level)),
                unit_keys=tuple(_split_ampersand(row.get("unit_list") or row.get("units_list"))),
                avg_placement=float(row["avg"]) if row.get("avg") is not None else None,
                count=int(row.get("count", 0)),
                score=float(row.get("win", 0.0)),
                source="metatft_comp_details_early_options",
            )
        )
    for level_key, rows in _option_groups(details.get("options")).items():
        row = _best_option(rows, prefer="score")
        if row is None:
            continue
        level = int(level_key)
        line = "mid"
        stage, stage_round = 3, 2
        if level >= 8:
            line, stage, stage_round = "late", 4, 5
        if level >= 9:
            line, stage, stage_round = "final", 5, 5
        lines.append(
            CompStageLine(
                line=f"{line}_{level_key}",
                stage=stage,
                stage_round=stage_round,
                level=max(1, min(10, level)),
                unit_keys=tuple(_split_ampersand(row.get("units_list") or row.get("unit_list"))),
                trait_keys=tuple(
                    _trait_key_from_token(token)
                    for token in _split_ampersand(row.get("traits_list"))
                    if _trait_key_from_token(token)
                ),
                avg_placement=float(row["avg"]) if row.get("avg") is not None else None,
                count=int(row.get("count", 0)),
                score=float(row.get("score", 0.0)),
                source="metatft_comp_details_options",
            )
        )
    lines.sort(key=lambda line: (line.stage, line.stage_round, line.level, line.line))
    return tuple(lines)


def _iter_option_rows(details: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for group in (details.get("options"), details.get("early_options")):
        for option_rows in _option_groups(group).values():
            rows.extend(option_rows)
    return rows


def _option_groups(group: object) -> dict[str, list[dict[str, Any]]]:
    if not isinstance(group, dict):
        return {}
    out: dict[str, list[dict[str, Any]]] = {}
    for level, rows in group.items():
        if isinstance(rows, list):
            out[str(level)] = [row for row in rows if isinstance(row, dict)]
    return out


def _best_option(rows: list[dict[str, Any]], *, prefer: str) -> dict[str, Any] | None:
    if not rows:
        return None
    if prefer == "win":
        return max(
            rows,
            key=lambda row: (
                float(row.get("win", 0.0)),
                int(row.get("count", 0)),
                -float(row.get("avg", 99.0)),
            ),
        )
    return max(
        rows,
        key=lambda row: (
            float(row.get("score", 0.0)),
            int(row.get("count", 0)),
            -float(row.get("avg", 99.0)),
        ),
    )


def _split_ampersand(value: object) -> list[str]:
    if not value:
        return []
    if isinstance(value, list | tuple):
        return [str(part) for part in value if part]
    return [part for part in str(value).split("&") if part]


def _trait_key_from_token(token: str) -> str:
    return re.sub(r"_\d+$", "", str(token))


def _trait_level_from_token(token: str) -> int | None:
    match = re.search(r"_(\d+)$", str(token))
    return int(match.group(1)) if match else None


def _display_name(key: str) -> str:
    name = re.sub(r"^TFT\d+_", "", key)
    name = name.replace("_", " ")
    return re.sub(r"(?<=[a-z])(?=[A-Z])", " ", name).strip()

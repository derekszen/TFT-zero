"""Encoders for current-patch MetaTFT board and comp-line states."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from mini_tft.metatft.catalog import MetaTFTCatalog, MetaTFTComp
from mini_tft.metatft.schema import (
    MAX_BENCH_TOKENS,
    MAX_BOARD_TOKENS,
    MAX_ITEMS_PER_UNIT,
    CurrentBoardState,
    CurrentBoardUnit,
)

DEFAULT_MAX_TRAITS = 16
DEFAULT_MAX_AUGMENTS = 6

STAGE_LINE_SPECS: tuple[tuple[str, int, int, int], ...] = (
    ("early", 2, 1, 4),
    ("mid", 3, 2, 6),
    ("late", 4, 5, 8),
    ("final", 5, 5, 10),
)


@dataclass(frozen=True)
class EncodedBoardState:
    unit_namespace: str
    scalars: NDArray[np.float32]
    board_unit_ids: NDArray[np.int16]
    board_stars: NDArray[np.int8]
    board_item_ids: NDArray[np.int16]
    board_positions: NDArray[np.int8]
    bench_unit_ids: NDArray[np.int16]
    active_trait_ids: NDArray[np.int16]
    augment_ids: NDArray[np.int16]
    target_comp_id: np.int16


@dataclass(frozen=True)
class EncodedStageLine:
    line: str
    state: CurrentBoardState
    encoding: EncodedBoardState


class CurrentBoardEncoder:
    """Numpy encoder for symbolic current-patch board states."""

    def __init__(
        self,
        catalog: MetaTFTCatalog,
        *,
        max_board_tokens: int = MAX_BOARD_TOKENS,
        max_bench_tokens: int = MAX_BENCH_TOKENS,
        max_items_per_unit: int = MAX_ITEMS_PER_UNIT,
        max_traits: int = DEFAULT_MAX_TRAITS,
        max_augments: int = DEFAULT_MAX_AUGMENTS,
        include_target_stats: bool = True,
        include_target_comp_id: bool = True,
    ) -> None:
        self.catalog = catalog
        self.max_board_tokens = max_board_tokens
        self.max_bench_tokens = max_bench_tokens
        self.max_items_per_unit = max_items_per_unit
        self.max_traits = max_traits
        self.max_augments = max_augments
        self.include_target_stats = include_target_stats
        self.include_target_comp_id = include_target_comp_id

    @property
    def scalar_dim(self) -> int:
        return 11

    def encode(self, state: CurrentBoardState) -> EncodedBoardState:
        if len(state.board) > self.max_board_tokens:
            raise ValueError("board has more units than this encoder supports")
        if len(state.bench) > self.max_bench_tokens:
            raise ValueError("bench has more units than this encoder supports")

        target_comp = (
            self.catalog.comp(state.target_comp_id)
            if state.target_comp_id and self.include_target_stats
            else None
        )
        return EncodedBoardState(
            unit_namespace=self.catalog.unit_namespace,
            scalars=self._scalars(state, target_comp),
            board_unit_ids=self._unit_ids(state.board, self.max_board_tokens),
            board_stars=self._stars(state.board, self.max_board_tokens),
            board_item_ids=self._item_ids(state.board, self.max_board_tokens),
            board_positions=self._positions(state.board, self.max_board_tokens),
            bench_unit_ids=self._unit_ids(state.bench, self.max_bench_tokens),
            active_trait_ids=self._tag_ids(state.active_trait_keys, self.max_traits),
            augment_ids=self._augment_ids(state.augment_keys, self.max_augments),
            target_comp_id=np.int16(
                self.catalog.comp_index(state.target_comp_id)
                if self.include_target_comp_id
                else 0
            ),
        )

    def encode_final_board(self, comp_id: str) -> EncodedBoardState:
        return self.encode(final_board_state(self.catalog, comp_id))

    def encode_stage_lines(self, comp_id: str) -> tuple[EncodedStageLine, ...]:
        states = derive_stage_line_states(self.catalog, comp_id)
        return tuple(
            EncodedStageLine(
                line=str(state.metadata["line"]),
                state=state,
                encoding=self.encode(state),
            )
            for state in states
        )

    def _scalars(
        self,
        state: CurrentBoardState,
        target_comp: MetaTFTComp | None,
    ) -> NDArray[np.float32]:
        target_rank = target_comp.rank / max(1, self.catalog.comp_count) if target_comp else 0.0
        target_avg = target_comp.avg_placement / 8.0 if target_comp else 0.0
        target_top4 = target_comp.top4_rate if target_comp else 0.0
        target_win = target_comp.win_rate if target_comp else 0.0
        values = [
            state.stage / 8.0,
            state.stage_round / 7.0,
            state.level / 10.0,
            state.gold / 100.0,
            state.hp / 100.0,
            len(state.board) / self.max_board_tokens,
            len(state.bench) / self.max_bench_tokens,
            len(state.active_trait_keys) / max(1, self.max_traits),
            target_rank,
            target_avg,
            target_top4 + target_win,
        ]
        return np.asarray(values, dtype=np.float32)

    def _unit_ids(
        self,
        units: tuple[CurrentBoardUnit, ...],
        width: int,
    ) -> NDArray[np.int16]:
        ids = np.zeros((width,), dtype=np.int16)
        for index, unit in enumerate(units[:width]):
            ids[index] = self.catalog.unit_id(unit.unit_key)
        return ids

    def _stars(
        self,
        units: tuple[CurrentBoardUnit, ...],
        width: int,
    ) -> NDArray[np.int8]:
        stars = np.zeros((width,), dtype=np.int8)
        for index, unit in enumerate(units[:width]):
            stars[index] = unit.stars
        return stars

    def _item_ids(
        self,
        units: tuple[CurrentBoardUnit, ...],
        width: int,
    ) -> NDArray[np.int16]:
        item_ids = np.zeros((width, self.max_items_per_unit), dtype=np.int16)
        for unit_index, unit in enumerate(units[:width]):
            for item_index, item_key in enumerate(unit.item_keys[: self.max_items_per_unit]):
                item_ids[unit_index, item_index] = self.catalog.item_id(item_key)
        return item_ids

    def _positions(
        self,
        units: tuple[CurrentBoardUnit, ...],
        width: int,
    ) -> NDArray[np.int8]:
        positions = np.zeros((width,), dtype=np.int8)
        for index, unit in enumerate(units[:width]):
            positions[index] = min(unit.position, 127)
        return positions

    def _tag_ids(self, tag_keys: tuple[str, ...], width: int) -> NDArray[np.int16]:
        ids = np.zeros((width,), dtype=np.int16)
        for index, tag_key in enumerate(tag_keys[:width]):
            ids[index] = self.catalog.tag_id(tag_key)
        return ids

    def _augment_ids(self, augment_keys: tuple[str, ...], width: int) -> NDArray[np.int16]:
        ids = np.zeros((width,), dtype=np.int16)
        for index, augment_key in enumerate(augment_keys[:width]):
            ids[index] = self.catalog.augment_id(augment_key)
        return ids


def final_board_state(catalog: MetaTFTCatalog, comp_id: str) -> CurrentBoardState:
    comp = catalog.comp(comp_id)
    item_builds = _best_item_builds_by_unit(comp)
    return _state_from_comp_units(
        comp,
        line="final",
        stage=5,
        stage_round=5,
        level=comp.level_hint,
        max_units=MAX_BOARD_TOKENS,
        source="metatft_final_board",
        unit_item_keys=item_builds,
    )


def derive_stage_line_states(
    catalog: MetaTFTCatalog,
    comp_id: str,
) -> tuple[CurrentBoardState, ...]:
    comp = catalog.comp(comp_id)
    if comp.stage_lines:
        item_builds = _best_item_builds_by_unit(comp)
        return tuple(
            CurrentBoardState(
                stage=line.stage,
                stage_round=line.stage_round,
                level=line.level,
                board=tuple(
                    CurrentBoardUnit(
                        unit_key=unit_key,
                        position=position,
                        item_keys=item_builds.get(unit_key, ()),
                    )
                    for position, unit_key in enumerate(line.unit_keys[:MAX_BOARD_TOKENS])
                ),
                active_trait_keys=line.trait_keys or _trait_breakpoint_keys(comp),
                augment_keys=comp.augment_keys,
                target_comp_id=comp.comp_id,
                source=line.source,
                metadata={
                    "line": line.line,
                    "comp_rank": comp.rank,
                    "avg_placement": line.avg_placement
                    if line.avg_placement is not None
                    else comp.avg_placement,
                    "top4_rate": comp.top4_rate,
                    "win_rate": comp.win_rate,
                    "count": line.count,
                    "score": line.score,
                },
            )
            for line in comp.stage_lines
        )
    return tuple(
        _state_from_comp_units(
            comp,
            line=line,
            stage=stage,
            stage_round=stage_round,
            level=min(max_units, comp.level_hint),
            max_units=min(max_units, MAX_BOARD_TOKENS),
            source="metatft_final_board_projection",
        )
        for line, stage, stage_round, max_units in STAGE_LINE_SPECS
    )


def _state_from_comp_units(
    comp: MetaTFTComp,
    *,
    line: str,
    stage: int,
    stage_round: int,
    level: int,
    max_units: int,
    source: str,
    unit_item_keys: dict[str, tuple[str, ...]] | None = None,
) -> CurrentBoardState:
    item_map = unit_item_keys or {}
    units = tuple(
        CurrentBoardUnit(
            unit_key=unit_key,
            position=position,
            item_keys=item_map.get(unit_key, ()),
        )
        for position, unit_key in enumerate(comp.unit_keys[:max_units])
    )
    return CurrentBoardState(
        stage=stage,
        stage_round=stage_round,
        level=max(1, min(10, level)),
        board=units,
        active_trait_keys=_trait_tags(comp) or _trait_breakpoint_keys(comp),
        augment_keys=comp.augment_keys or _augment_tags(comp),
        target_comp_id=comp.comp_id,
        source=source,
        metadata={
            "line": line,
            "comp_rank": comp.rank,
            "avg_placement": comp.avg_placement,
            "top4_rate": comp.top4_rate,
            "win_rate": comp.win_rate,
        },
    )


def _trait_tags(comp: MetaTFTComp) -> tuple[str, ...]:
    return tuple(tag for tag in comp.tag_keys if "Trait" in tag)


def _augment_tags(comp: MetaTFTComp) -> tuple[str, ...]:
    return tuple(tag for tag in comp.tag_keys if "Augment" in tag)


def _trait_breakpoint_keys(comp: MetaTFTComp) -> tuple[str, ...]:
    return tuple(dict.fromkeys(bp.trait_key for bp in comp.trait_breakpoints))


def _best_item_builds_by_unit(comp: MetaTFTComp) -> dict[str, tuple[str, ...]]:
    builds: dict[str, tuple[str, ...]] = {}
    for build in comp.item_builds:
        if build.unit_key not in builds:
            builds[build.unit_key] = build.item_keys
    return builds

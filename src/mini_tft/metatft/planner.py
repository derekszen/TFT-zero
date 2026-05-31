"""Planner scoring utilities for current-patch MetaTFT board states."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from numpy.typing import NDArray

from mini_tft.metatft.catalog import MetaTFTCatalog
from mini_tft.metatft.encoders import CurrentBoardEncoder, EncodedBoardState
from mini_tft.metatft.schema import (
    MAX_BENCH_TOKENS,
    MAX_BOARD_TOKENS,
    CurrentBoardState,
    CurrentBoardUnit,
)
from mini_tft.metatft.value_training import CurrentPatchValueConfig, CurrentPatchValueNet

RankBy = Literal["after_value", "delta"]


@dataclass(frozen=True)
class CandidateTransition:
    """One candidate shop/bench/board transition for a planner to score."""

    action: str
    before: CurrentBoardState
    after: CurrentBoardState
    metadata: Mapping[str, str | int | float] = field(default_factory=dict)


@dataclass(frozen=True)
class ScoredTransition:
    rank: int
    action: str
    after_value: float
    before_value: float
    delta: float
    rank_score: float
    transition: CandidateTransition


class CurrentPatchPlannerScorer:
    """Ranks symbolic board transitions with a trained current-patch value model."""

    def __init__(
        self,
        catalog: MetaTFTCatalog,
        model: CurrentPatchValueNet,
        *,
        device_name: str = "cpu",
        encoder: CurrentBoardEncoder | None = None,
    ) -> None:
        self.catalog = catalog
        self.device = torch.device(device_name)
        self.model = model.to(self.device)
        self.model.eval()
        self.encoder = encoder or CurrentBoardEncoder(catalog)

    @classmethod
    def from_checkpoint(
        cls,
        catalog: MetaTFTCatalog,
        checkpoint: Path,
        *,
        device_name: str = "cpu",
        blind_target_metadata: bool | None = None,
    ) -> CurrentPatchPlannerScorer:
        payload = torch.load(checkpoint, map_location=device_name)
        config = CurrentPatchValueConfig(**payload["config"])
        encoder_config = payload.get("encoder", {})
        include_target_stats = bool(encoder_config.get("include_target_stats", True))
        include_target_comp_id = bool(encoder_config.get("include_target_comp_id", True))
        if blind_target_metadata is not None:
            include_target_stats = not blind_target_metadata
            include_target_comp_id = not blind_target_metadata
        _validate_checkpoint_catalog(
            config,
            catalog,
            require_comp_ids=include_target_comp_id,
        )
        model = CurrentPatchValueNet(config)
        model.load_state_dict(payload["model_state"])
        encoder = CurrentBoardEncoder(
            catalog,
            include_target_stats=include_target_stats,
            include_target_comp_id=include_target_comp_id,
        )
        return cls(catalog, model, device_name=device_name, encoder=encoder)

    def score_states(self, states: Sequence[CurrentBoardState]) -> NDArray[np.float32]:
        if not states:
            return np.asarray([], dtype=np.float32)
        encodings = [self.encoder.encode(state) for state in states]
        with torch.no_grad():
            values = self.model(_encodings_to_tensors(encodings, self.device))
        return values.detach().cpu().numpy().astype(np.float32)

    def rank_transitions(
        self,
        transitions: Sequence[CandidateTransition],
        *,
        rank_by: RankBy = "after_value",
    ) -> tuple[ScoredTransition, ...]:
        if rank_by not in {"after_value", "delta"}:
            raise ValueError("rank_by must be 'after_value' or 'delta'")
        if not transitions:
            return ()
        before_states = [transition.before for transition in transitions]
        after_states = [transition.after for transition in transitions]
        before_values = self.score_states(before_states)
        after_values = self.score_states(after_states)
        scored = []
        for index, transition in enumerate(transitions):
            delta = float(after_values[index] - before_values[index])
            rank_score = float(after_values[index]) if rank_by == "after_value" else delta
            scored.append(
                ScoredTransition(
                    rank=0,
                    action=transition.action,
                    after_value=float(after_values[index]),
                    before_value=float(before_values[index]),
                    delta=delta,
                    rank_score=rank_score,
                    transition=transition,
                )
            )
        scored.sort(key=lambda row: row.rank_score, reverse=True)
        return tuple(replace(row, rank=index + 1) for index, row in enumerate(scored))


def build_shop_bench_board_transitions(
    state: CurrentBoardState,
    *,
    shop_unit_keys: Sequence[str] = (),
    unit_costs: Mapping[str, int] | None = None,
    default_unit_cost: int | None = 3,
    include_hold: bool = True,
    include_buy_to_board: bool = True,
    include_swaps: bool = True,
) -> tuple[CandidateTransition, ...]:
    """Build simple one-action candidates from shop, bench, and board zones."""

    transitions: list[CandidateTransition] = []
    if include_hold:
        transitions.append(_transition("hold", state, state, {"type": "hold"}))

    for shop_index, unit_key in enumerate(shop_unit_keys):
        if not unit_key:
            continue
        cost = _unit_cost(unit_key, unit_costs, default_unit_cost)
        if cost is not None and state.gold < cost:
            continue
        if len(state.bench) < MAX_BENCH_TOKENS:
            unit = CurrentBoardUnit(unit_key=unit_key)
            after = _replace_state(
                state,
                bench=(*state.bench, unit),
                gold=state.gold - (cost or 0),
                action=f"buy_shop_{shop_index}_to_bench",
            )
            transitions.append(
                _transition(
                    f"buy_shop_{shop_index}_to_bench:{unit_key}",
                    state,
                    after,
                    {"type": "buy_to_bench", "shop_index": shop_index, "unit_key": unit_key},
                )
            )
        if include_buy_to_board and _has_board_space(state):
            unit = CurrentBoardUnit(unit_key=unit_key, position=len(state.board))
            after = _replace_state(
                state,
                board=(*state.board, unit),
                gold=state.gold - (cost or 0),
                action=f"buy_shop_{shop_index}_to_board",
            )
            transitions.append(
                _transition(
                    f"buy_shop_{shop_index}_to_board:{unit_key}",
                    state,
                    after,
                    {"type": "buy_to_board", "shop_index": shop_index, "unit_key": unit_key},
                )
            )

    for bench_index, bench_unit in enumerate(state.bench):
        if _has_board_space(state):
            board_unit = replace(bench_unit, position=len(state.board))
            bench = tuple(
                unit for index, unit in enumerate(state.bench) if index != bench_index
            )
            after = _replace_state(
                state,
                board=(*state.board, board_unit),
                bench=bench,
                action=f"field_bench_{bench_index}",
            )
            transitions.append(
                _transition(
                    f"field_bench_{bench_index}:{bench_unit.unit_key}",
                    state,
                    after,
                    {
                        "type": "field_bench",
                        "bench_index": bench_index,
                        "unit_key": bench_unit.unit_key,
                    },
                )
            )
        if include_swaps:
            for board_index, board_unit in enumerate(state.board):
                board = list(state.board)
                bench = list(state.bench)
                board[board_index] = replace(bench_unit, position=board_unit.position)
                bench[bench_index] = replace(board_unit, position=bench_unit.position)
                after = _replace_state(
                    state,
                    board=tuple(board),
                    bench=tuple(bench),
                    action=f"swap_bench_{bench_index}_board_{board_index}",
                )
                transitions.append(
                    _transition(
                        f"swap_bench_{bench_index}_board_{board_index}:"
                        f"{bench_unit.unit_key}<->{board_unit.unit_key}",
                        state,
                        after,
                        {
                            "type": "swap",
                            "bench_index": bench_index,
                            "board_index": board_index,
                            "bench_unit_key": bench_unit.unit_key,
                            "board_unit_key": board_unit.unit_key,
                        },
                    )
                )
    return tuple(transitions)


def _encodings_to_tensors(
    encodings: Sequence[EncodedBoardState],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    return {
        "scalars": torch.as_tensor(
            np.stack([encoding.scalars for encoding in encodings]).astype(np.float32),
            device=device,
        ),
        "board_unit_ids": torch.as_tensor(
            np.stack([encoding.board_unit_ids for encoding in encodings]).astype(np.int64),
            device=device,
        ),
        "board_item_ids": torch.as_tensor(
            np.stack([encoding.board_item_ids for encoding in encodings]).astype(np.int64),
            device=device,
        ),
        "active_trait_ids": torch.as_tensor(
            np.stack([encoding.active_trait_ids for encoding in encodings]).astype(np.int64),
            device=device,
        ),
        "augment_ids": torch.as_tensor(
            np.stack([encoding.augment_ids for encoding in encodings]).astype(np.int64),
            device=device,
        ),
        "target_comp_ids": torch.as_tensor(
            np.asarray([encoding.target_comp_id for encoding in encodings], dtype=np.int64),
            device=device,
        ),
    }


def _validate_checkpoint_catalog(
    config: CurrentPatchValueConfig,
    catalog: MetaTFTCatalog,
    *,
    require_comp_ids: bool,
) -> None:
    if catalog.unit_count > config.max_unit_id:
        raise ValueError("checkpoint does not support this catalog's unit IDs")
    if catalog.item_count > config.max_item_id:
        raise ValueError("checkpoint does not support this catalog's item IDs")
    if catalog.tag_count > config.max_trait_id:
        raise ValueError("checkpoint does not support this catalog's trait/tag IDs")
    if catalog.augment_count > config.max_augment_id:
        raise ValueError("checkpoint does not support this catalog's augment IDs")
    if require_comp_ids and catalog.comp_count > config.max_comp_id:
        raise ValueError("checkpoint does not support this catalog's comp IDs")


def _transition(
    action: str,
    before: CurrentBoardState,
    after: CurrentBoardState,
    metadata: Mapping[str, str | int | float],
) -> CandidateTransition:
    return CandidateTransition(action=action, before=before, after=after, metadata=metadata)


def _replace_state(
    state: CurrentBoardState,
    *,
    action: str,
    board: tuple[CurrentBoardUnit, ...] | None = None,
    bench: tuple[CurrentBoardUnit, ...] | None = None,
    gold: int | None = None,
) -> CurrentBoardState:
    metadata = {**state.metadata, "planner_action": action}
    return replace(
        state,
        board=state.board if board is None else board,
        bench=state.bench if bench is None else bench,
        gold=state.gold if gold is None else max(0, gold),
        source="planner_candidate",
        metadata=metadata,
    )


def _has_board_space(state: CurrentBoardState) -> bool:
    return len(state.board) < min(state.level, MAX_BOARD_TOKENS)


def _unit_cost(
    unit_key: str,
    unit_costs: Mapping[str, int] | None,
    default_unit_cost: int | None,
) -> int | None:
    cost = unit_costs.get(unit_key) if unit_costs is not None else None
    if cost is None:
        return default_unit_cost
    return max(0, int(cost))

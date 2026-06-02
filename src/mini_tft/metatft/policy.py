"""Shop/economy policy shell for current-patch MetaTFT board planning."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Protocol

from mini_tft.metatft.catalog import MetaTFTCatalog
from mini_tft.metatft.metrics import target_comp_units_for_level
from mini_tft.metatft.planner import (
    CandidateTransition,
    CurrentPatchPlannerScorer,
    RankBy,
    ScoredTransition,
    build_shop_bench_board_transitions,
)
from mini_tft.metatft.schema import (
    MAX_BOARD_TOKENS,
    MAX_LEVEL,
    CurrentBoardState,
    CurrentBoardUnit,
)


class TransitionScorer(Protocol):
    def rank_transitions(
        self,
        transitions: Sequence[CandidateTransition],
        *,
        rank_by: RankBy = "after_value",
    ) -> tuple[ScoredTransition, ...]:
        """Rank candidate transitions from best to worst."""


@dataclass(frozen=True)
class ShopEconPolicyConfig:
    roll_cost: int = 2
    xp_buy_cost: int = 4
    xp_level_increment: int = 1
    max_level: int = MAX_LEVEL
    max_actions_per_turn: int = 12
    min_value_delta: float = 0.02
    roll_min_stage: int = 4
    roll_gold_reserve: int = 20
    xp_gold_reserve: int = 8
    default_unit_cost: int = 3
    target_completion_weight: float = 10.0
    target_extra_penalty: float = 2.0
    target_duplicate_penalty: float = 2.0
    enable_target_refill: bool = True
    roll_for_missing_targets: bool = True


@dataclass(frozen=True)
class PolicyTurnPlan:
    decisions: tuple[ScoredTransition, ...]
    final_state: CurrentBoardState
    final_shop: tuple[str, ...]
    stopped: bool


class CurrentPatchShopEconPolicy:
    """Uses a value scorer for board actions with explicit TFT-style econ pacing."""

    def __init__(
        self,
        scorer: TransitionScorer,
        *,
        config: ShopEconPolicyConfig | None = None,
        unit_costs: Mapping[str, int] | None = None,
        catalog: MetaTFTCatalog | None = None,
    ) -> None:
        self.scorer = scorer
        self.config = config or ShopEconPolicyConfig()
        self.unit_costs = dict(unit_costs or {})
        self.catalog = catalog

    @classmethod
    def from_checkpoint(
        cls,
        catalog: MetaTFTCatalog,
        checkpoint: Path,
        *,
        device_name: str = "cpu",
        blind_target_metadata: bool | None = None,
        config: ShopEconPolicyConfig | None = None,
    ) -> CurrentPatchShopEconPolicy:
        scorer = CurrentPatchPlannerScorer.from_checkpoint(
            catalog,
            checkpoint,
            device_name=device_name,
            blind_target_metadata=blind_target_metadata,
        )
        return cls(
            scorer,
            config=config,
            unit_costs=_catalog_unit_costs(catalog),
            catalog=catalog,
        )

    def choose_action(
        self,
        state: CurrentBoardState,
        *,
        shop_unit_keys: Sequence[str] = (),
        unit_costs: Mapping[str, int] | None = None,
        rank_by: RankBy = "after_value",
    ) -> ScoredTransition:
        costs = self.unit_costs if unit_costs is None else unit_costs
        candidates = [
            *build_shop_bench_board_transitions(
                state,
                shop_unit_keys=shop_unit_keys,
                unit_costs=costs,
                default_unit_cost=self.config.default_unit_cost,
                include_hold=False,
            ),
            *self._target_refill_transitions(state),
            *self._econ_transitions(state),
        ]
        ranked = self._rank_transitions(candidates, rank_by=rank_by)
        if not ranked:
            raise ValueError("policy produced no legal candidate transitions")

        if self._target_board_is_exact(state):
            end_turn = next(
                (candidate for candidate in ranked if self._action_type(candidate) == "end_turn"),
                None,
            )
            if end_turn is not None:
                return end_turn

        if self._should_roll_for_missing_targets(state, shop_unit_keys):
            roll = next(
                (candidate for candidate in ranked if self._action_type(candidate) == "roll"),
                None,
            )
            if roll is not None:
                return roll

        best = ranked[0]
        if self._action_type(best) in {"buy_xp", "roll"}:
            return best
        if self._action_type(best) == "end_turn":
            return best
        if best.delta >= self.config.min_value_delta or self._beats_end_turn(best, ranked):
            return best
        fallback = next(
            (candidate for candidate in ranked if self._action_type(candidate) == "end_turn"),
            ranked[-1],
        )
        return fallback

    def plan_turn(
        self,
        state: CurrentBoardState,
        *,
        shops: Sequence[Sequence[str]],
        unit_costs: Mapping[str, int] | None = None,
        rank_by: RankBy = "after_value",
    ) -> PolicyTurnPlan:
        current_state = state
        shop_cursor = 0
        current_shop = list(shops[0]) if shops else []
        decisions: list[ScoredTransition] = []
        stopped = False

        for _ in range(self.config.max_actions_per_turn):
            decision = self.choose_action(
                current_state,
                shop_unit_keys=current_shop,
                unit_costs=unit_costs,
                rank_by=rank_by,
            )
            decisions.append(decision)
            action_type = self._action_type(decision)
            current_state = decision.transition.after
            if action_type == "end_turn":
                stopped = True
                break
            if action_type == "roll":
                shop_cursor += 1
                current_shop = list(shops[shop_cursor]) if shop_cursor < len(shops) else []
                continue
            shop_index = decision.transition.metadata.get("shop_index")
            if action_type in {"buy_to_bench", "buy_to_board"} and isinstance(shop_index, int):
                if 0 <= shop_index < len(current_shop):
                    current_shop[shop_index] = ""

        return PolicyTurnPlan(
            decisions=tuple(decisions),
            final_state=current_state,
            final_shop=tuple(current_shop),
            stopped=stopped,
        )

    def _econ_transitions(self, state: CurrentBoardState) -> tuple[CandidateTransition, ...]:
        transitions = [
            CandidateTransition(
                action="end_turn",
                before=state,
                after=replace(
                    state,
                    source="policy_candidate",
                    metadata={**state.metadata, "policy_action": "end_turn"},
                ),
                metadata={"type": "end_turn"},
            )
        ]
        if self._can_buy_xp(state):
            transitions.append(
                CandidateTransition(
                    action="buy_xp",
                    before=state,
                    after=replace(
                        state,
                        level=min(
                            self.config.max_level,
                            state.level + self.config.xp_level_increment,
                        ),
                        gold=max(0, state.gold - self.config.xp_buy_cost),
                        source="policy_candidate",
                        metadata={**state.metadata, "policy_action": "buy_xp"},
                    ),
                    metadata={"type": "buy_xp"},
                )
            )
        if self._can_roll(state):
            transitions.append(
                CandidateTransition(
                    action="roll",
                    before=state,
                    after=replace(
                        state,
                        gold=max(0, state.gold - self.config.roll_cost),
                        source="policy_candidate",
                        metadata={**state.metadata, "policy_action": "roll"},
                    ),
                    metadata={"type": "roll"},
                )
            )
        return tuple(transitions)

    def _target_refill_transitions(
        self,
        state: CurrentBoardState,
    ) -> tuple[CandidateTransition, ...]:
        if not self.config.enable_target_refill:
            return ()
        target_units = self._target_units(state)
        if not target_units:
            return ()
        board, bench = _best_target_refill(state, target_units)
        if board == state.board and bench == state.bench:
            return ()
        return (
            CandidateTransition(
                action="target_refill_board",
                before=state,
                after=replace(
                    state,
                    board=board,
                    bench=bench,
                    source="policy_candidate",
                    metadata={**state.metadata, "policy_action": "target_refill_board"},
                ),
                metadata={"type": "target_refill_board"},
            ),
        )

    def _rank_transitions(
        self,
        candidates: Sequence[CandidateTransition],
        *,
        rank_by: RankBy,
    ) -> tuple[ScoredTransition, ...]:
        ranked = self.scorer.rank_transitions(candidates, rank_by=rank_by)
        if not ranked or self.catalog is None or self.config.target_completion_weight <= 0:
            return ranked
        adjusted = []
        for row in ranked:
            target_score = self._target_board_score(row.transition.after)
            adjusted.append(
                replace(
                    row,
                    rank_score=row.rank_score
                    + self.config.target_completion_weight * target_score,
                )
            )
        adjusted.sort(key=lambda row: row.rank_score, reverse=True)
        return tuple(replace(row, rank=index + 1) for index, row in enumerate(adjusted))

    def _target_board_score(self, state: CurrentBoardState) -> float:
        target_units = self._target_units(state)
        if not target_units:
            return 0.0
        board_counts = Counter(state.board_unit_keys)
        target_counts = Counter(target_units)
        overlap = sum((board_counts & target_counts).values())
        missing = sum((target_counts - board_counts).values())
        extra = sum((board_counts - target_counts).values())
        duplicate_excess = sum(
            max(0, count - target_counts.get(unit_key, 0))
            for unit_key, count in board_counts.items()
        )
        return (
            float(overlap)
            - self.config.target_extra_penalty * float(extra + missing)
            - self.config.target_duplicate_penalty * float(duplicate_excess)
        )

    def _target_units(self, state: CurrentBoardState) -> tuple[str, ...]:
        if self.catalog is None or state.target_comp_id is None:
            return ()
        try:
            comp = self.catalog.comp(state.target_comp_id)
        except KeyError:
            return ()
        return target_comp_units_for_level(comp, state.level)

    def _should_roll_for_missing_targets(
        self,
        state: CurrentBoardState,
        shop_unit_keys: Sequence[str],
    ) -> bool:
        if not self.config.roll_for_missing_targets or not self._can_roll(state):
            return False
        missing = self._missing_owned_target_units(state)
        if not missing:
            return False
        missing_counts = Counter(missing)
        return not any(missing_counts.get(unit_key, 0) > 0 for unit_key in shop_unit_keys)

    def _missing_owned_target_units(self, state: CurrentBoardState) -> tuple[str, ...]:
        target_units = self._target_units(state)
        if not target_units:
            return ()
        target_counts = Counter(target_units)
        owned_counts = Counter((*state.board_unit_keys, *(unit.unit_key for unit in state.bench)))
        return tuple((target_counts - owned_counts).elements())

    def _target_board_is_exact(self, state: CurrentBoardState) -> bool:
        target_units = self._target_units(state)
        if not target_units:
            return False
        return Counter(state.board_unit_keys) == Counter(target_units)

    def _beats_end_turn(
        self,
        best: ScoredTransition,
        ranked: Sequence[ScoredTransition],
    ) -> bool:
        end_turn = next(
            (candidate for candidate in ranked if self._action_type(candidate) == "end_turn"),
            None,
        )
        if end_turn is None:
            return False
        return best.rank_score > end_turn.rank_score + self.config.min_value_delta

    def _can_buy_xp(self, state: CurrentBoardState) -> bool:
        if state.level >= self.config.max_level or state.gold < self.config.xp_buy_cost:
            return False
        desired_level = _desired_level_for_stage(state.stage)
        if state.level < desired_level:
            return state.gold - self.config.xp_buy_cost >= 0
        return state.gold - self.config.xp_buy_cost >= self.config.xp_gold_reserve + 40

    def _can_roll(self, state: CurrentBoardState) -> bool:
        if state.stage < self.config.roll_min_stage or state.gold < self.config.roll_cost:
            return False
        return state.gold - self.config.roll_cost >= self.config.roll_gold_reserve

    @staticmethod
    def _action_type(decision: ScoredTransition) -> str:
        action_type = decision.transition.metadata.get("type")
        return str(action_type) if action_type is not None else "unknown"


def _desired_level_for_stage(stage: int) -> int:
    if stage <= 2:
        return 4
    if stage == 3:
        return 6
    if stage == 4:
        return 8
    return 9


def _catalog_unit_costs(catalog: MetaTFTCatalog) -> dict[str, int]:
    return {
        unit.key: unit.cost
        for unit in catalog.units
        if unit.cost is not None and unit.cost > 0
    }


def _best_target_refill(
    state: CurrentBoardState,
    target_units: Sequence[str],
) -> tuple[tuple[CurrentBoardUnit, ...], tuple[CurrentBoardUnit, ...]]:
    owned = [*state.board, *state.bench]
    selected_indexes: set[int] = set()
    board: list[CurrentBoardUnit] = []
    board_limit = min(state.level, MAX_BOARD_TOKENS, len(target_units))

    for target_unit in target_units[:board_limit]:
        selected_index = _first_unused_unit_index(owned, selected_indexes, target_unit)
        if selected_index is None:
            continue
        selected_indexes.add(selected_index)
        board.append(replace(owned[selected_index], position=len(board)))

    for index, unit in enumerate(owned):
        if len(board) >= min(state.level, MAX_BOARD_TOKENS):
            break
        if index in selected_indexes:
            continue
        selected_indexes.add(index)
        board.append(replace(unit, position=len(board)))

    bench = tuple(
        replace(unit, position=0)
        for index, unit in enumerate(owned)
        if index not in selected_indexes
    )
    return tuple(board), bench


def _first_unused_unit_index(
    units: Sequence[CurrentBoardUnit],
    used_indexes: set[int],
    unit_key: str,
) -> int | None:
    for index, unit in enumerate(units):
        if index not in used_indexes and unit.unit_key == unit_key:
            return index
    return None

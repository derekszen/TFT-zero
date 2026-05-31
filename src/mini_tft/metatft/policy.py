"""Shop/economy policy shell for current-patch MetaTFT board planning."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Protocol

from mini_tft.metatft.catalog import MetaTFTCatalog
from mini_tft.metatft.planner import (
    CandidateTransition,
    CurrentPatchPlannerScorer,
    RankBy,
    ScoredTransition,
    build_shop_bench_board_transitions,
)
from mini_tft.metatft.schema import MAX_LEVEL, CurrentBoardState


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
    ) -> None:
        self.scorer = scorer
        self.config = config or ShopEconPolicyConfig()
        self.unit_costs = dict(unit_costs or {})

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
        return cls(scorer, config=config, unit_costs=_catalog_unit_costs(catalog))

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
            *self._econ_transitions(state),
        ]
        ranked = self.scorer.rank_transitions(candidates, rank_by=rank_by)
        if not ranked:
            raise ValueError("policy produced no legal candidate transitions")

        best = ranked[0]
        if self._action_type(best) in {"buy_xp", "roll"}:
            return best
        if self._action_type(best) == "end_turn":
            return best
        if best.delta >= self.config.min_value_delta:
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

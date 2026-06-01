from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from mini_tft.metatft import (
    CandidateTransition,
    CatalogAugment,
    CatalogItem,
    CurrentBoardEncoder,
    CurrentBoardState,
    CurrentBoardUnit,
    CurrentPatchPlannerScorer,
    CurrentPatchShopEconPolicy,
    PlannerMetricRequirement,
    PolicyTurnPlan,
    ScoredTransition,
    ShopEconPolicyConfig,
    build_shop_bench_board_transitions,
    derive_stage_line_states,
    evaluate_planner_batch_gate,
    evaluate_planner_trace_batch,
    final_board_state,
    load_catalog_from_comp_strength,
    load_catalog_from_payload,
    top_comp_match_report,
)
from mini_tft.metatft.catalog import UNIT_NAMESPACE
from mini_tft.metatft.schema import MAX_BOARD_TOKENS
from mini_tft.metatft.value_training import (
    build_value_training_batch,
    train_current_patch_value_model,
)

FIXTURE = Path("tests/fixtures/metatft_set17_comp_strength_2026-05-31.json")


def test_load_catalog_normalizes_current_patch_fixture() -> None:
    catalog = load_catalog_from_comp_strength(FIXTURE)

    assert catalog.metadata.tft_set == "TFTSet17"
    assert catalog.metadata.patch == "current"
    assert catalog.metadata.cluster_id == "409"
    assert catalog.unit_namespace == UNIT_NAMESPACE
    assert catalog.comp_count == 16
    assert catalog.unit_count >= 48
    assert catalog.tag_count >= 20
    assert catalog.units[0].unit_id == 1
    assert catalog.unit_id("TFT17_Aatrox") > 0
    assert catalog.comp("409003").rank == 1
    assert catalog.comp("409003").avg_placement < catalog.comp("409068").avg_placement

    assert catalog._unit_by_key["TFT17_MissFortune"].display_name == "Miss Fortune"
    assert catalog._tag_by_key["TFT17_ASTrait"].kind == "trait"
    assert catalog._tag_by_key["TFT17_Augment_JaxCarry"].kind == "augment"
    assert catalog._tag_by_key["TFT17_MissFortune"].kind == "unit"


def test_current_board_state_schema_validates_current_patch_bounds() -> None:
    state = CurrentBoardState(
        stage=3,
        stage_round=2,
        level=6,
        gold=42,
        hp=88,
        board=(CurrentBoardUnit("TFT17_Aatrox", stars=2, position=7),),
        bench=(CurrentBoardUnit("TFT17_Maokai"),),
        active_trait_keys=("TFT17_ASTrait",),
        augment_keys=("TFT17_Augment_JaxCarry",),
        target_comp_id="409003",
    )

    assert state.stage_label == "3-2"
    assert state.board_unit_keys == ("TFT17_Aatrox",)

    with pytest.raises(ValueError, match="level"):
        CurrentBoardState(stage=1, stage_round=1, level=11, board=())

    with pytest.raises(ValueError, match="board tokens"):
        CurrentBoardState(
            stage=1,
            stage_round=1,
            level=10,
            board=tuple(CurrentBoardUnit("TFT17_Aatrox") for _ in range(MAX_BOARD_TOKENS + 1)),
        )


def test_final_board_encoder_preserves_units_duplicates_and_padding() -> None:
    catalog = load_catalog_from_comp_strength(FIXTURE)
    state = final_board_state(catalog, "409003")
    encoder = CurrentBoardEncoder(catalog)
    encoded = encoder.encode(state)

    ornn_id = catalog.unit_id("TFT17_Ornn")
    assert state.source == "metatft_final_board"
    assert state.target_comp_id == "409003"
    assert state.level == 9
    assert state.board_unit_keys.count("TFT17_Ornn") == 2
    assert encoded.unit_namespace == UNIT_NAMESPACE
    assert encoded.scalars.shape == (encoder.scalar_dim,)
    assert encoded.scalars.dtype == np.float32
    assert encoded.board_unit_ids.shape == (MAX_BOARD_TOKENS,)
    assert encoded.board_stars.shape == (MAX_BOARD_TOKENS,)
    assert encoded.board_item_ids.shape == (MAX_BOARD_TOKENS, 3)
    assert encoded.bench_unit_ids.shape == (12,)
    assert encoded.active_trait_ids.shape == (16,)
    assert int(encoded.target_comp_id) == 1
    assert np.count_nonzero(encoded.board_unit_ids) == 9
    assert np.count_nonzero(encoded.board_unit_ids == ornn_id) == 2
    assert np.all(encoded.board_unit_ids[9:] == 0)


def test_encoder_supports_manual_state_with_items_traits_augments_and_bench() -> None:
    catalog = load_catalog_from_comp_strength(FIXTURE)
    catalog = catalog.__class__(
        metadata=catalog.metadata,
        units=catalog.units,
        tags=catalog.tags,
        comps=catalog.comps,
        items=(CatalogItem(1, "TFT_Item_Rageblade", "Rageblade"),),
        augments=(CatalogAugment(1, "TFT17_Augment_JaxCarry", "Jax Carry"),),
    )
    state = CurrentBoardState(
        stage=4,
        stage_round=2,
        level=8,
        gold=30,
        hp=72,
        board=(
            CurrentBoardUnit(
                "TFT17_Jax",
                stars=2,
                item_keys=("TFT_Item_Rageblade",),
                position=3,
            ),
        ),
        bench=(CurrentBoardUnit("TFT17_Aatrox"),),
        active_trait_keys=("TFT17_ASTrait",),
        augment_keys=("TFT17_Augment_JaxCarry",),
        target_comp_id="409000",
    )

    encoded = CurrentBoardEncoder(catalog).encode(state)

    assert encoded.board_unit_ids[0] == catalog.unit_id("TFT17_Jax")
    assert encoded.board_stars[0] == 2
    assert encoded.board_item_ids[0, 0] == 1
    assert encoded.board_positions[0] == 3
    assert encoded.bench_unit_ids[0] == catalog.unit_id("TFT17_Aatrox")
    assert encoded.active_trait_ids[0] == catalog.tag_id("TFT17_ASTrait")
    assert encoded.augment_ids[0] == 1


def test_encoder_can_blind_target_comp_metadata_for_heldout_validation() -> None:
    catalog = load_catalog_from_comp_strength(FIXTURE)
    state = final_board_state(catalog, "409003")

    leaked = CurrentBoardEncoder(catalog).encode(state)
    blinded = CurrentBoardEncoder(
        catalog,
        include_target_stats=False,
        include_target_comp_id=False,
    ).encode(state)

    assert int(leaked.target_comp_id) == catalog.comp_index("409003")
    assert int(blinded.target_comp_id) == 0
    assert np.count_nonzero(leaked.scalars[-3:]) > 0
    assert np.count_nonzero(blinded.scalars[-3:]) == 0


def test_stage_line_encoder_projects_final_comp_into_early_mid_late_final_states() -> None:
    catalog = load_catalog_from_comp_strength(FIXTURE)
    encoder = CurrentBoardEncoder(catalog)

    states = derive_stage_line_states(catalog, "409003")
    encoded_lines = encoder.encode_stage_lines("409003")

    assert [state.metadata["line"] for state in states] == ["early", "mid", "late", "final"]
    assert [line.line for line in encoded_lines] == ["early", "mid", "late", "final"]
    assert [len(line.state.board) for line in encoded_lines] == [4, 6, 8, 9]
    assert [line.state.stage_label for line in encoded_lines] == ["2-1", "3-2", "4-5", "5-5"]
    assert all(line.state.source == "metatft_final_board_projection" for line in encoded_lines)
    assert all(
        line.encoding.board_unit_ids[0] == catalog.unit_id("TFT17_Aatrox")
        for line in encoded_lines
    )
    assert np.count_nonzero(encoded_lines[0].encoding.board_unit_ids) == 4
    assert np.count_nonzero(encoded_lines[-1].encoding.board_unit_ids) == 9


def test_rich_catalog_ingests_real_item_trait_augment_and_line_shapes() -> None:
    catalog = load_catalog_from_payload(_rich_fixture_payload())
    comp = catalog.comp("409003")
    final_state = final_board_state(catalog, "409003")
    encoded_lines = CurrentBoardEncoder(catalog).encode_stage_lines("409003")

    assert catalog.item_count == 3
    assert catalog._unit_by_key["TFT17_MissFortune"].cost == 4
    assert catalog.augment_id("TFT17_Augment_JaxCarry") > 0
    assert catalog.trait_count >= 2
    assert comp.item_builds[0].unit_key == "TFT17_MissFortune"
    assert comp.trait_breakpoints[0].trait_key == "TFT17_ASTrait"
    assert any(trait.key == "TFT17_ASTrait" and trait.breakpoints for trait in catalog.traits)
    assert final_state.board_unit_keys.count("TFT17_MissFortune") == 1
    assert final_state.board[2].item_keys == (
        "TFT_Item_Deathblade",
        "TFT_Item_GuinsoosRageblade",
        "TFT_Item_InfinityEdge",
    )
    assert [line.line for line in encoded_lines] == ["early_4", "late_8"]
    assert encoded_lines[0].state.source == "metatft_comp_details_early_options"
    assert encoded_lines[1].state.active_trait_keys == ("TFT17_ASTrait", "TFT17_DRX")


def test_current_patch_value_training_batch_uses_encoder_outputs() -> None:
    catalog = load_catalog_from_payload(_rich_fixture_payload())

    batch = build_value_training_batch(catalog, include_stage_lines=True)

    assert batch.scalars.shape[0] == len(batch.targets)
    assert batch.board_unit_ids.shape[0] == len(batch.targets)
    assert batch.board_item_ids.shape[2] == 3
    assert "early_4" in batch.lines
    assert "late_8" in batch.lines
    assert np.count_nonzero(batch.board_item_ids) > 0
    assert batch.targets[0] == pytest.approx(-4.0216)


def test_current_patch_value_model_training_smoke(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    assert torch is not None
    catalog = load_catalog_from_payload(_rich_fixture_payload())
    checkpoint = tmp_path / "current_patch_value.pt"

    report = train_current_patch_value_model(
        catalog,
        output=checkpoint,
        device_name="cpu",
        epochs=5,
        learning_rate=1e-3,
        hidden_dim=32,
        embed_dim=16,
    )

    assert checkpoint.exists()
    assert report.examples >= 4
    assert report.loss >= 0.0
    assert 0.0 <= report.pairwise_accuracy <= 1.0


def test_current_patch_value_training_reports_heldout_comp_rankings(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    catalog = load_catalog_from_comp_strength(FIXTURE)
    checkpoint = tmp_path / "current_patch_value_heldout.pt"

    report = train_current_patch_value_model(
        catalog,
        output=checkpoint,
        device_name="cpu",
        epochs=5,
        learning_rate=1e-3,
        hidden_dim=32,
        embed_dim=16,
        validation_fraction=0.25,
        blind_target_metadata=True,
    )
    payload = torch.load(checkpoint, map_location="cpu")

    assert report.train_examples < report.examples
    assert report.heldout_examples > 0
    assert report.heldout_comp_count > 0
    assert report.heldout_pairwise_accuracy is not None
    assert report.heldout_spearman is not None
    assert report.target_metadata_blinded is True
    assert payload["encoder"] == {
        "include_target_stats": False,
        "include_target_comp_id": False,
    }
    assert payload["metrics"]["heldout_examples"] == report.heldout_examples


def test_current_patch_planner_scorer_ranks_shop_bench_board_transitions(
    tmp_path: Path,
) -> None:
    pytest.importorskip("torch")
    catalog = load_catalog_from_payload(_rich_fixture_payload())
    checkpoint = tmp_path / "current_patch_value_planner.pt"
    train_current_patch_value_model(
        catalog,
        output=checkpoint,
        device_name="cpu",
        epochs=8,
        learning_rate=1e-3,
        hidden_dim=32,
        embed_dim=16,
        blind_target_metadata=True,
    )
    scorer = CurrentPatchPlannerScorer.from_checkpoint(
        catalog,
        checkpoint,
        device_name="cpu",
    )
    state = CurrentBoardState(
        stage=3,
        stage_round=2,
        level=4,
        gold=10,
        board=(CurrentBoardUnit("TFT17_Aatrox", position=0),),
        bench=(CurrentBoardUnit("TFT17_MissFortune", position=0),),
        active_trait_keys=("TFT17_ASTrait",),
        target_comp_id="409003",
    )
    candidates = build_shop_bench_board_transitions(
        state,
        shop_unit_keys=("TFT17_Belveth", "TFT17_Ornn"),
        unit_costs={"TFT17_Belveth": 2, "TFT17_Ornn": 4},
    )

    ranked = scorer.rank_transitions(candidates)

    assert len(ranked) >= 5
    assert [row.rank for row in ranked] == list(range(1, len(ranked) + 1))
    assert all(np.isfinite(row.after_value) for row in ranked)
    assert all(
        ranked[index].rank_score >= ranked[index + 1].rank_score
        for index in range(len(ranked) - 1)
    )
    assert any(row.action.startswith("field_bench_0") for row in ranked)
    assert any(row.transition.after.source == "planner_candidate" for row in ranked)


def test_current_patch_shop_econ_policy_loops_shop_and_board_actions() -> None:
    policy = CurrentPatchShopEconPolicy(
        _TypePriorityScorer(),
        config=ShopEconPolicyConfig(max_actions_per_turn=2, min_value_delta=-1.0),
    )
    state = CurrentBoardState(
        stage=3,
        stage_round=2,
        level=3,
        gold=12,
        board=(CurrentBoardUnit("TFT17_Aatrox", position=0),),
        bench=(CurrentBoardUnit("TFT17_MissFortune", position=0),),
        active_trait_keys=("TFT17_ASTrait",),
        target_comp_id="409003",
    )

    plan = policy.plan_turn(
        state,
        shops=(("TFT17_Belveth", "TFT17_Ornn"),),
        unit_costs={"TFT17_Belveth": 2, "TFT17_Ornn": 4},
    )

    action_types = [decision.transition.metadata["type"] for decision in plan.decisions]
    assert action_types[:2] == ["field_bench", "buy_to_board"]
    assert plan.final_state.gold == 10
    assert len(plan.final_state.board) == 3
    assert plan.final_shop[0] == ""


def test_top_comp_match_report_matches_level_9_final_board() -> None:
    catalog = load_catalog_from_comp_strength(FIXTURE)
    state = final_board_state(catalog, "409003")

    report = top_comp_match_report(
        catalog,
        state.board_unit_keys,
        board_level=9,
        levels=(9,),
        top_k=3,
    )

    match = report[0]
    assert match.level == 9
    assert match.eligible is True
    assert match.comp_id == "409003"
    assert match.comp_rank == 1
    assert match.exact_match is True
    assert match.partial_match is True
    assert match.overlap_count == match.target_unit_count == len(state.board)
    assert match.jaccard == pytest.approx(1.0)
    assert match.missing_units == ()
    assert match.extra_units == ()


def test_top_comp_match_report_uses_rich_level_8_stage_line() -> None:
    catalog = load_catalog_from_payload(_rich_fixture_payload())
    comp = catalog.comp("409003")
    level_8_line = next(line for line in comp.stage_lines if line.level == 8)

    report = top_comp_match_report(
        catalog,
        level_8_line.unit_keys,
        board_level=8,
        levels=(8,),
        top_k=1,
    )

    match = report[0]
    assert match.level == 8
    assert match.eligible is True
    assert match.comp_id == "409003"
    assert match.exact_match is True
    assert match.target_unit_count == 8
    assert match.overlap_count == 8


def test_top_comp_match_report_marks_underleveled_traces_ineligible() -> None:
    catalog = load_catalog_from_comp_strength(FIXTURE)
    state = final_board_state(catalog, "409003")

    report = top_comp_match_report(
        catalog,
        state.board_unit_keys[:6],
        board_level=6,
        levels=(8, 9),
        top_k=3,
    )

    assert [match.level for match in report] == [8, 9]
    assert all(match.eligible is False for match in report)
    assert all(match.partial_match is False for match in report)
    assert all(match.exact_match is False for match in report)


def test_planner_trace_batch_summarizes_level_8_and_9_match_rates() -> None:
    catalog = load_catalog_from_comp_strength(FIXTURE)
    report = evaluate_planner_trace_batch(
        catalog,
        _FinalCompPlanner(catalog),
        comp_ids=("409003", "409000"),
        demo_levels=(8, 9),
        match_levels=(8, 9),
        top_k=16,
        min_recall=0.75,
    )

    assert report.comp_ids == ("409003", "409000")
    assert len(report.traces) == 4
    assert [summary.level for summary in report.summaries] == [8, 9]

    level_8 = report.summaries[0]
    assert level_8.trace_count == 4
    assert level_8.eligible_count == 4
    assert level_8.exact_match_count == _count_matches(report.traces, 8, "exact_match")
    assert level_8.good_enough_count == 4
    assert level_8.good_enough_rate == pytest.approx(1.0)

    level_9 = report.summaries[1]
    assert level_9.trace_count == 4
    assert level_9.eligible_count == 2
    assert level_9.exact_match_count == _count_matches(report.traces, 9, "exact_match")
    assert level_9.good_enough_count == _count_matches(report.traces, 9, "good_enough")
    assert level_9.good_enough_rate == pytest.approx(0.5)
    assert level_9.eligible_good_enough_rate == pytest.approx(1.0)

    level_9_failures = report.exact_failure_summaries[1]
    assert level_9_failures.level == 9
    assert level_9_failures.failed_count == level_9.trace_count - level_9.exact_match_count
    assert level_9_failures.underleveled_count == 2
    assert level_9_failures.examples


def test_planner_batch_gate_reports_threshold_failures() -> None:
    catalog = load_catalog_from_comp_strength(FIXTURE)
    report = evaluate_planner_trace_batch(
        catalog,
        _FinalCompPlanner(catalog),
        comp_ids=("409003", "409000"),
        demo_levels=(8, 9),
        match_levels=(8, 9),
        top_k=16,
        min_recall=0.75,
    )

    passing_gate = evaluate_planner_batch_gate(
        report,
        (
            PlannerMetricRequirement(
                level=8,
                metric="good_enough_rate",
                minimum=1.0,
            ),
            PlannerMetricRequirement(
                level=9,
                metric="eligible_good_enough_rate",
                minimum=1.0,
            ),
        ),
    )
    assert passing_gate.passed is True
    assert passing_gate.failures == ()

    failing_gate = evaluate_planner_batch_gate(
        report,
        (
            PlannerMetricRequirement(
                level=9,
                metric="good_enough_rate",
                minimum=0.75,
            ),
        ),
    )
    assert failing_gate.passed is False
    assert failing_gate.failures[0].level == 9
    assert failing_gate.failures[0].actual == pytest.approx(0.5)


class _TypePriorityScorer:
    priorities = {
        "field_bench": 100.0,
        "buy_to_board": 90.0,
        "buy_xp": 80.0,
        "buy_to_bench": 70.0,
        "swap": 60.0,
        "end_turn": 0.0,
        "roll": -10.0,
    }

    def rank_transitions(
        self,
        transitions: list[CandidateTransition],
        *,
        rank_by: str = "after_value",
    ) -> tuple[ScoredTransition, ...]:
        del rank_by
        scored = []
        for transition in transitions:
            action_type = str(transition.metadata.get("type", "unknown"))
            score = self.priorities.get(action_type, -100.0)
            scored.append(
                ScoredTransition(
                    rank=0,
                    action=transition.action,
                    after_value=score,
                    before_value=0.0,
                    delta=score,
                    rank_score=score,
                    transition=transition,
                )
            )
        scored.sort(key=lambda row: row.rank_score, reverse=True)
        return tuple(
            ScoredTransition(
                rank=index + 1,
                action=row.action,
                after_value=row.after_value,
                before_value=row.before_value,
                delta=row.delta,
                rank_score=row.rank_score,
                transition=row.transition,
            )
            for index, row in enumerate(scored)
        )


class _FinalCompPlanner:
    def __init__(self, catalog) -> None:
        self.catalog = catalog

    def plan_turn(
        self,
        state: CurrentBoardState,
        *,
        shops,
        unit_costs=None,
        rank_by: str = "after_value",
    ) -> PolicyTurnPlan:
        del shops, unit_costs, rank_by
        comp = self.catalog.comp(state.target_comp_id)
        final_state = replace(
            state,
            board=tuple(
                CurrentBoardUnit(unit_key=unit_key, position=index)
                for index, unit_key in enumerate(comp.unit_keys[: state.level])
            ),
            bench=(),
        )
        return PolicyTurnPlan(decisions=(), final_state=final_state, final_shop=(), stopped=True)


def _count_matches(traces, level: int, field: str) -> int:
    return sum(
        1
        for trace in traces
        for match in trace.matches
        if match.level == level and getattr(match, field)
    )


def _fixture_source_records() -> tuple[dict[str, object], list[dict[str, object]]]:
    import json

    payload = json.loads(FIXTURE.read_text(encoding="utf-8"))
    return payload["source"], payload["records"]


def _rich_fixture_payload() -> dict[str, object]:
    source, records = _fixture_source_records()
    return {
        "source": source,
        "records": records[:2],
        "rich": {
            "comp_details": {
                "409003": {
                    "results": {
                        "builds": [
                            {
                                "unit": "TFT17_MissFortune",
                                "buildName": [
                                    "TFT_Item_Deathblade",
                                    "TFT_Item_GuinsoosRageblade",
                                    "TFT_Item_InfinityEdge",
                                ],
                                "avg": 4.2472,
                                "count": 16636,
                                "score": 0.4707,
                                "place_change": -0.197,
                            }
                        ],
                        "traits": [
                            {
                                "trait": "TFT17_ASTrait",
                                "levels": [
                                    {"level": 1, "count": 36789, "avg": 4.7987},
                                    {"level": 2, "count": 17685, "avg": 3.9003},
                                ],
                            }
                        ],
                        "early_options": {
                            "4": [
                                {
                                    "unit_list": (
                                        "TFT17_Belveth&TFT17_Briar&"
                                        "TFT17_MissFortune&TFT17_RekSai"
                                    ),
                                    "count": 1213,
                                    "level": 4.047,
                                    "avg": 3.9143,
                                    "win": 0.5235,
                                }
                            ]
                        },
                        "options": {
                            "8": [
                                {
                                    "units_list": (
                                        "TFT17_Aatrox&TFT17_Belveth&TFT17_Kindred&"
                                        "TFT17_Maokai&TFT17_MissFortune&TFT17_Ornn&"
                                        "TFT17_Rhaast&TFT17_Urgot"
                                    ),
                                    "traits_list": "TFT17_ASTrait_2&TFT17_DRX_1",
                                    "score": 95.995,
                                    "avg": 3.1771,
                                    "count": 6398,
                                }
                            ]
                        },
                    }
                }
            },
            "comp_builds": {"results": {}},
            "comp_options": {"results": {"options": {}}},
            "unit_items_processed": {
                "itemNames": [
                    "TFT_Item_Deathblade",
                    "TFT_Item_GuinsoosRageblade",
                    "TFT_Item_InfinityEdge",
                ],
                "units": {},
            },
            "stat_items": {
                "results": [
                    {
                        "itemName": "TFT_Item_GuinsoosRageblade",
                        "places": [100, 90, 80, 70, 60, 50, 40, 30],
                    }
                ]
            },
            "unit_costs": {
                "TFT17_Aatrox": 1,
                "TFT17_Belveth": 2,
                "TFT17_Kindred": 3,
                "TFT17_Maokai": 1,
                "TFT17_MissFortune": 4,
                "TFT17_Ornn": 4,
                "TFT17_Rhaast": 5,
                "TFT17_RekSai": 1,
                "TFT17_Urgot": 5,
            },
            "tables": {
                "itemEffects": {
                    "TFT17_Augment_JaxCarry": {"tier": "Tier2"},
                }
            },
        },
    }

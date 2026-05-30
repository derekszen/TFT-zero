from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from mini_tft.metatft import (
    CatalogAugment,
    CatalogItem,
    CurrentBoardEncoder,
    CurrentBoardState,
    CurrentBoardUnit,
    derive_stage_line_states,
    final_board_state,
    load_catalog_from_comp_strength,
    load_catalog_from_payload,
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
            "tables": {
                "itemEffects": {
                    "TFT17_Augment_JaxCarry": {"tier": "Tier2"},
                }
            },
        },
    }

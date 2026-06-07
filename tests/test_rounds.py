from __future__ import annotations

from mini_tft.core.rounds import is_item_drop_round, round_info


def test_set1_round_schedule_maps_opening_and_pve_rounds() -> None:
    cases = {
        1: ("Stage 1-1", "pve"),
        2: ("Stage 1-2", "pve"),
        3: ("Stage 1-3", "pve"),
        4: ("Stage 2-1", "pvp"),
        10: ("Stage 2-7", "pve"),
        11: ("Stage 3-1", "pvp"),
        17: ("Stage 3-7", "pve"),
    }

    for round_num, (label, round_type) in cases.items():
        current_round = round_info(round_num)
        assert current_round.stage_label == label
        assert current_round.round_type == round_type
        assert is_item_drop_round(round_num) is (round_type == "pve")


def test_round_schedule_rejects_non_positive_rounds() -> None:
    try:
        round_info(0)
    except ValueError as error:
        assert "round_num must be positive" in str(error)
    else:
        raise AssertionError("round_info accepted round 0")

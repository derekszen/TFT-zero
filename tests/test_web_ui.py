from __future__ import annotations

from mini_tft.core.actions import Action
from mini_tft.core.state import UnitInstance
from mini_tft.web.server import MiniTFTWebSession


def test_web_session_payload_exposes_interactive_state() -> None:
    session = MiniTFTWebSession(seed=11)

    payload = session.payload()

    assert payload["status"]["round"] == 1
    assert payload["status"]["stage_label"] == "Stage 1-1"
    assert payload["enemy"]["label"] == "Stage 1-1 enemy"
    assert len(payload["enemy"]["slots"]) >= 2
    assert len(payload["shop"]) == 5
    assert len(payload["board"]) == 9
    assert len(payload["bench"]) == 9
    assert payload["actions"][Action.END_TURN]["legal"] is True


def test_web_session_can_step_and_reset() -> None:
    session = MiniTFTWebSession(seed=12)

    after_step = session.step(Action.END_TURN)
    assert after_step["status"]["round"] == 2
    assert after_step["last"]["action"] == Action.END_TURN
    assert after_step["last"]["legal"] is True

    after_reset = session.reset(seed=13)
    assert after_reset["seed"] == 13
    assert after_reset["status"]["round"] == 1
    assert after_reset["log"][0] == "Reset seed 13"


def test_web_session_can_manually_move_unit_from_bench_to_board() -> None:
    session = MiniTFTWebSession(seed=14)
    assert session.env.state is not None
    session.env.state.bench[0] = UnitInstance(unit_id=1)

    payload = session.move_unit("bench", 0, "board", 0)

    assert payload["last"]["legal"] is True
    assert payload["status"]["step_count"] == 0
    assert payload["board"][0]["id"] == 1
    assert payload["bench"][0] is None


def test_web_session_manual_move_respects_board_cap_but_allows_swaps() -> None:
    session = MiniTFTWebSession(seed=15)
    assert session.env.state is not None
    session.env.state.level = 1
    session.env.state.board[0] = UnitInstance(unit_id=1)
    session.env.state.bench[0] = UnitInstance(unit_id=2)

    failed = session.move_unit("bench", 0, "board", 1)
    assert failed["last"]["legal"] is False
    assert failed["board"][1] is None
    assert failed["bench"][0]["id"] == 2

    swapped = session.move_unit("bench", 0, "board", 0)
    assert swapped["last"]["legal"] is True
    assert swapped["board"][0]["id"] == 2
    assert swapped["bench"][0]["id"] == 1

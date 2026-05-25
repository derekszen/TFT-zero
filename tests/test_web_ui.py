from __future__ import annotations

from mini_tft.core.actions import Action
from mini_tft.web.server import MiniTFTWebSession


def test_web_session_payload_exposes_interactive_state() -> None:
    session = MiniTFTWebSession(seed=11)

    payload = session.payload()

    assert payload["status"]["round"] == 1
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

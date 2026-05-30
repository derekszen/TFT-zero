from __future__ import annotations

import pytest

pytest.importorskip("torch")

import torch

from mini_tft.core.actions import Action
from mini_tft.core.config import EnvConfig
from mini_tft.core.env import MiniTFTEnv
from mini_tft.core.state import UnitInstance
from mini_tft.fight_model.model import FightModelConfig, FightValueNet


def test_fight_value_combat_model_can_drive_env_end_turn(tmp_path) -> None:
    checkpoint = tmp_path / "fight_value.pt"
    model_config = FightModelConfig(max_unit_id=32, max_item_id=16)
    model = FightValueNet(model_config)
    torch.save(
        {
            "model_state": model.state_dict(),
            "config": model_config.__dict__,
        },
        checkpoint,
    )

    env = MiniTFTEnv(
        EnvConfig(
            seed=123,
            combat_model="fight_value",
            fight_value_checkpoint=str(checkpoint),
            fight_value_device="cpu",
        )
    )
    env.reset()
    state = env.state
    assert state is not None
    state.board[0] = UnitInstance(unit_id=1)
    state.board[6] = UnitInstance(unit_id=2, items=[1])

    _, _, _, _, info = env.step(Action.END_TURN)

    assert info["combat_model"] == "fight_value"
    assert info["board_strength"] >= 0.0
    assert state.last_enemy_strength > 0.0
    assert 0.0 <= state.last_board_strength


def test_fight_value_config_requires_checkpoint() -> None:
    with pytest.raises(ValueError, match="fight_value_checkpoint"):
        MiniTFTEnv(EnvConfig(combat_model="fight_value"))


def test_metatft_checkpoint_is_rejected_by_mini_tft_env(tmp_path) -> None:
    checkpoint = tmp_path / "metatft_value.pt"
    model_config = FightModelConfig(max_unit_id=32, max_item_id=1)
    model = FightValueNet(model_config)
    torch.save(
        {
            "model_state": model.state_dict(),
            "config": model_config.__dict__,
            "metatft_unit_id_lookup": {"TFT17_Aatrox": 1},
            "unit_namespace": "metatft_current_patch",
        },
        checkpoint,
    )

    with pytest.raises(ValueError, match="same unit namespace"):
        MiniTFTEnv(
            EnvConfig(
                combat_model="fight_value",
                fight_value_checkpoint=str(checkpoint),
                fight_value_device="cpu",
            )
        )

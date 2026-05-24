from __future__ import annotations

from mini_tft.core.config import EnvConfig
from mini_tft.core.economy import apply_xp, income_after_combat, interest


def test_interest_caps_at_five() -> None:
    assert interest(0) == 0
    assert interest(39) == 3
    assert interest(50) == 5
    assert interest(90) == 5


def test_income_uses_base_interest_and_win_gold() -> None:
    config = EnvConfig()

    assert income_after_combat(30, won=False, config=config) == 8
    assert income_after_combat(30, won=True, config=config) == 9


def test_apply_xp_levels_repeatedly() -> None:
    level, xp = apply_xp(level=3, xp=0, amount=20, max_level=9)

    assert level == 5
    assert xp == 4

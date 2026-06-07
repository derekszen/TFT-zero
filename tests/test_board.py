from __future__ import annotations

from collections import Counter

from mini_tft.core.board import generate_candidate_boards
from mini_tft.core.combat import board_strength
from mini_tft.core.config import EnvConfig
from mini_tft.core.ids import EMPTY
from mini_tft.core.set_data import load_set
from mini_tft.core.state import GameState, UnitInstance, new_game_state, state_signature


def test_candidate_boards_are_ranked_and_improve_when_stronger_board_exists() -> None:
    config = EnvConfig(seed=21, starting_level=2)
    data = load_set(config.dataset)
    state = _state(config)
    state.level = 2
    state.board[0] = UnitInstance(unit_id=1)
    state.board[1] = UnitInstance(unit_id=6)
    state.bench[0] = UnitInstance(unit_id=20)
    state.bench[1] = UnitInstance(unit_id=23)

    current_strength = board_strength(state.board, data).strength

    candidates = generate_candidate_boards(state, data, config, top_k=4)

    assert candidates
    assert [candidate.strength for candidate in candidates] == sorted(
        (candidate.strength for candidate in candidates),
        reverse=True,
    )
    assert candidates[0].strength > current_strength
    assert _unit_ids(candidates[0].board) == [20, 23]


def test_candidate_boards_respect_level_cap() -> None:
    config = EnvConfig(seed=22, starting_level=2)
    data = load_set(config.dataset)
    state = _state(config)
    state.level = 2
    state.board[0] = UnitInstance(unit_id=1)
    state.bench[0] = UnitInstance(unit_id=2)
    state.bench[1] = UnitInstance(unit_id=13)
    state.bench[2] = UnitInstance(unit_id=20)
    state.bench[3] = UnitInstance(unit_id=23)

    candidates = generate_candidate_boards(state, data, config, top_k=20)

    assert candidates
    for candidate in candidates:
        assert len(candidate.board) == config.max_level
        assert sum(unit is not None for unit in candidate.board) <= state.level


def test_candidate_board_generation_does_not_mutate_state() -> None:
    config = EnvConfig(seed=23, starting_level=3)
    data = load_set(config.dataset)
    state = _state(config)
    state.level = 3
    state.board[0] = UnitInstance(unit_id=1, stars=2, items=[6])
    state.board[6] = UnitInstance(unit_id=2, items=[1])
    state.bench[0] = UnitInstance(unit_id=24, items=[4])
    before = state_signature(state)

    candidates = generate_candidate_boards(state, data, config, top_k=3)

    assert state_signature(state) == before
    candidate_unit = next(unit for unit in candidates[0].board if unit is not None)
    candidate_unit.items.append(10)
    assert state_signature(state) == before


def test_candidate_boards_preserve_owned_unit_multiset_without_duplicates() -> None:
    config = EnvConfig(seed=24, starting_level=3)
    data = load_set(config.dataset)
    state = _state(config)
    state.level = 3
    state.board[0] = UnitInstance(unit_id=1, stars=2, items=[6])
    state.board[6] = UnitInstance(unit_id=2, items=[1])
    state.bench[0] = UnitInstance(unit_id=2, items=[1])
    state.bench[1] = UnitInstance(unit_id=24, items=[4, 10])
    owned_units = _unit_multiset([*state.board, *state.bench])

    candidates = generate_candidate_boards(state, data, config, top_k=10)

    assert candidates
    for candidate in candidates:
        candidate_units = _unit_multiset([*candidate.board, *candidate.bench])
        assert candidate_units == owned_units
        assert len(candidate.bench) == config.bench_size


def _state(config: EnvConfig) -> GameState:
    return new_game_state(config, seed=config.seed or 0, shop=[EMPTY] * config.shop_size)


def _unit_ids(units: tuple[UnitInstance | None, ...]) -> list[int]:
    return [unit.unit_id for unit in units if unit is not None]


def _unit_multiset(units: list[UnitInstance | None]) -> Counter[tuple[int, int, tuple[int, ...]]]:
    return Counter(
        (unit.unit_id, unit.stars, tuple(unit.items))
        for unit in units
        if unit is not None
    )

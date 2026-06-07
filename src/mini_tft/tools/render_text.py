"""Text renderer for debugging episodes."""

from __future__ import annotations

from mini_tft.core.combat import board_strength
from mini_tft.core.economy import xp_needed
from mini_tft.core.env import MiniTFTEnv
from mini_tft.core.ids import EMPTY
from mini_tft.core.rounds import round_info
from mini_tft.core.traits import active_trait_effects


def render_state(env: MiniTFTEnv) -> str:
    if env.state is None:
        return "MiniTFTEnv(uninitialized)"
    state = env.state
    current_round = round_info(state.round)
    lines = [
        f"{current_round.stage_label} ({current_round.round_type.upper()}) | "
        f"Round {state.round} | HP {state.hp} | Gold {state.gold} | "
        f"Level {state.level} | XP {state.xp}/{xp_needed(state.level)}",
        "",
        "Shop:",
    ]
    for index, unit_id in enumerate(state.shop):
        if unit_id == EMPTY:
            label = "empty"
        else:
            unit = env.data.units[unit_id]
            label = f"{unit.name} {unit.cost}g"
        lines.append(f"  [{index}] {label}")

    lines.append("")
    lines.append("Board:")
    for unit in state.board:
        if unit is None:
            continue
        unit_def = env.data.units[unit.unit_id]
        stars = "*" * unit.stars
        item_names = ", ".join(env.data.items[item_id].name for item_id in unit.items) or "none"
        lines.append(
            f"  {unit_def.name}{stars}  {unit_def.role}  "
            f"{' '.join(unit_def.traits)}  items: {item_names}"
        )

    lines.append("")
    bench = [
        "empty" if unit is None else f"{env.data.units[unit.unit_id].name}{'*' * unit.stars}"
        for unit in state.bench
    ]
    lines.append("Bench:")
    lines.append(f"  {' | '.join(bench)}")

    lines.append("")
    lines.append("Traits:")
    active = active_trait_effects(state.board, env.data)
    for trait, effects in active.items():
        lines.append(f"  {trait}: {effects}")

    stats = board_strength(state.board, env.data)
    enemy_index = min(state.round - 1, len(env.data.enemy_curve) - 1)
    lines.append("")
    lines.append(f"Strength: {stats.strength:.1f}")
    lines.append(f"Enemy next: {env.data.enemy_curve[enemy_index]:.1f}")
    return "\n".join(lines)


def main() -> None:
    env = MiniTFTEnv()
    env.reset(seed=0)
    print(render_state(env))


if __name__ == "__main__":
    main()

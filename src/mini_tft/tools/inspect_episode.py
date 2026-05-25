"""Run, watch, or interact with one simulator episode."""

from __future__ import annotations

import argparse
import time

import numpy as np

from mini_tft.bots import GreedyBoardBot, RandomBot
from mini_tft.core.actions import action_name
from mini_tft.core.config import EnvConfig
from mini_tft.rl.gym_env import MiniTFTGymEnv


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--bot", choices=["random", "greedy"], default="greedy")
    parser.add_argument("--mode", choices=["bot", "interactive"], default="bot")
    parser.add_argument("--delay", type=float, default=0.6)
    parser.add_argument("--max-steps", type=int, default=80)
    args = parser.parse_args()

    bot = RandomBot() if args.bot == "random" else GreedyBoardBot()
    env = MiniTFTGymEnv(config=EnvConfig(seed=args.seed))
    rng = np.random.default_rng(args.seed)
    obs, _ = env.reset(seed=args.seed)
    terminated = truncated = False
    step = 0
    print(_screen(env, step, action=None, reward=None))
    while not (terminated or truncated) and step < args.max_steps:
        if args.mode == "interactive":
            action = _prompt_action(env)
        else:
            action = bot.act(env, obs, rng)
        obs, reward, terminated, truncated, _ = env.step(action)
        step += 1
        print(_screen(env, step, action=action, reward=reward))
        if args.mode == "bot" and args.delay > 0:
            time.sleep(args.delay)

    if step >= args.max_steps and not (terminated or truncated):
        print(f"Stopped after --max-steps={args.max_steps}.")


def _screen(env: MiniTFTGymEnv, step: int, action: int | None, reward: float | None) -> str:
    lines = ["=" * 72, f"Step {step}"]
    if action is not None and reward is not None:
        lines.append(f"Action: {action} ({action_name(action)}) | Reward: {reward:.3f}")
    lines.append(env.render())
    lines.append("=" * 72)
    return "\n".join(lines)


def _prompt_action(env: MiniTFTGymEnv) -> int:
    mask = env.action_masks()
    legal = [index for index, value in enumerate(mask) if value]
    print("Legal actions:")
    for index in legal:
        print(f"  {index:2d}: {action_name(index)}")
    while True:
        raw = input("Choose action number, action name, or q to quit > ").strip().lower()
        if raw in {"q", "quit", "exit"}:
            raise SystemExit(0)
        if raw.isdigit():
            action = int(raw)
            if action in legal:
                return action
        matches = [index for index in legal if action_name(index) == raw]
        if len(matches) == 1:
            return matches[0]
        print("Invalid action. Pick one of the listed legal actions.")


if __name__ == "__main__":
    main()

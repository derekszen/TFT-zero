"""Train PPO through MiniTFT's emulated or native PufferLib env surfaces."""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import time
from collections.abc import Sequence
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
import torch
from gymnasium import spaces
from numpy.typing import NDArray
from torch import nn
from torch.distributions import Categorical

from mini_tft.core.config import EnvConfig
from mini_tft.rl.lobby_env import LOBBY_POLICY_BY_NAME
from mini_tft.rl.puffer_env import (
    make_lobby_hero_native_puffer_env,
    make_lobby_hero_puffer_env,
    make_puffer_env,
    make_strategic_native_puffer_env,
    make_strategic_puffer_env,
)
from mini_tft.rl.train_ppo import parse_hidden_sizes
from mini_tft.strategic.core import StrategicConfig

PufferEnvKind = Literal["single_player", "lobby", "strategic"]
PufferBackend = Literal["emulated", "native"]
TrainingConfig = EnvConfig | StrategicConfig


class MaskedActorCritic(nn.Module):
    """Small discrete-action PPO policy with observation-tail action masks."""

    def __init__(self, observation_dim: int, action_dim: int, hidden_sizes: Sequence[int]) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        last_dim = observation_dim
        for hidden_size in hidden_sizes:
            layers.extend([nn.Linear(last_dim, hidden_size), nn.Tanh()])
            last_dim = hidden_size
        self.backbone = nn.Sequential(*layers)
        self.actor = nn.Linear(last_dim, action_dim)
        self.critic = nn.Linear(last_dim, 1)
        self.action_dim = action_dim

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.backbone(obs.float())
        return self.actor(hidden), self.critic(hidden).squeeze(-1)

    def get_action_and_value(
        self,
        obs: torch.Tensor,
        action: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        logits, value = self(obs)
        mask = action_mask_from_observation(obs, self.action_dim)
        masked_logits = logits.masked_fill(~mask, -1.0e9)
        dist = Categorical(logits=masked_logits)
        if action is None:
            action = dist.sample()
        return action, dist.log_prob(action), dist.entropy(), value


def action_mask_from_observation(obs: torch.Tensor, action_dim: int) -> torch.Tensor:
    mask = obs[..., -action_dim:] > 0.5
    if torch.any(mask, dim=-1).all():
        return mask
    fallback = torch.zeros_like(mask)
    fallback[..., 0] = True
    return torch.where(torch.any(mask, dim=-1, keepdim=True), mask, fallback)


def make_training_env(
    *,
    env_kind: PufferEnvKind,
    seed: int,
    config: TrainingConfig,
    players: int,
    max_actions_per_player: int | None,
    lobby_opponent_policy: str,
    allow_oracle_macro_actions: bool,
) -> Any:
    if env_kind == "single_player":
        if not isinstance(config, EnvConfig):
            raise TypeError("single_player env requires EnvConfig")
        return make_puffer_env(seed=seed, config=config)
    if env_kind == "strategic":
        if not isinstance(config, StrategicConfig):
            raise TypeError("strategic env requires StrategicConfig")
        return make_strategic_puffer_env(seed=seed, config=config)
    if env_kind == "lobby":
        if not isinstance(config, EnvConfig):
            raise TypeError("lobby env requires EnvConfig")
        return make_lobby_hero_puffer_env(
            seed=seed,
            config=config,
            player_count=players,
            opponent_policy=LOBBY_POLICY_BY_NAME[lobby_opponent_policy],
            max_actions_per_player=max_actions_per_player,
            allow_oracle_macro_actions=allow_oracle_macro_actions,
        )
    raise ValueError(f"unsupported env kind: {env_kind}")


class EmulatedTrainingEnvBatch:
    """Batch adapter for the current one-emulated-env-per-slot trainer path."""

    auto_resets = False

    def __init__(self, envs: Sequence[Any]) -> None:
        if not envs:
            raise ValueError("envs must not be empty")
        self.envs = list(envs)
        self.num_envs = len(self.envs)
        self.single_action_space = self.envs[0].single_action_space

    def reset(self, seed: int) -> tuple[NDArray[np.float32], list[dict[str, Any]]]:
        return reset_envs(self.envs, seed)

    def step(
        self,
        actions: NDArray[np.int64],
    ) -> tuple[
        NDArray[np.float32],
        NDArray[np.float32],
        NDArray[np.bool_],
        NDArray[np.bool_],
        list[dict[str, Any]],
    ]:
        observations: list[NDArray[np.float32]] = []
        rewards = np.zeros(self.num_envs, dtype=np.float32)
        terminals = np.zeros(self.num_envs, dtype=np.bool_)
        truncations = np.zeros(self.num_envs, dtype=np.bool_)
        infos: list[dict[str, Any]] = []
        for env_index, env in enumerate(self.envs):
            obs, reward, terminated, truncated, info = env.step(int(actions[env_index]))
            observations.append(np.asarray(obs, dtype=np.float32).reshape(-1))
            rewards[env_index] = float(reward)
            terminals[env_index] = bool(terminated)
            truncations[env_index] = bool(truncated)
            infos.append(info)
        return np.stack(observations).astype(np.float32), rewards, terminals, truncations, infos

    def reset_one(self, env_index: int, seed: int) -> tuple[NDArray[np.float32], dict[str, Any]]:
        obs, info = self.envs[env_index].reset(seed=seed)
        return np.asarray(obs, dtype=np.float32).reshape(-1), info

    def close(self) -> None:
        for env in self.envs:
            env.close()


class NativePufferTrainingEnvBatch:
    """Batch adapter for native PufferEnv with one agent per training slot."""

    auto_resets = True

    def __init__(self, env: Any) -> None:
        self.env = env
        self.num_envs = int(env.num_agents)
        self.single_action_space = env.single_action_space

    def reset(self, seed: int) -> tuple[NDArray[np.float32], list[dict[str, Any]]]:
        obs, infos = self.env.reset(seed=seed)
        return np.asarray(obs, dtype=np.float32), list(infos)

    def step(
        self,
        actions: NDArray[np.int64],
    ) -> tuple[
        NDArray[np.float32],
        NDArray[np.float32],
        NDArray[np.bool_],
        NDArray[np.bool_],
        list[dict[str, Any]],
    ]:
        obs, rewards, terminals, truncations, infos = self.env.step(actions)
        return (
            np.asarray(obs, dtype=np.float32),
            np.asarray(rewards, dtype=np.float32),
            np.asarray(terminals, dtype=np.bool_),
            np.asarray(truncations, dtype=np.bool_),
            list(infos),
        )

    def reset_one(self, env_index: int, seed: int) -> tuple[NDArray[np.float32], dict[str, Any]]:
        del env_index, seed
        raise RuntimeError("native Puffer backend auto-resets completed envs")

    def close(self) -> None:
        self.env.close()


def make_training_env_batch(
    *,
    env_kind: PufferEnvKind,
    puffer_backend: PufferBackend,
    seed: int,
    config: TrainingConfig,
    num_envs: int,
    players: int,
    max_actions_per_player: int | None,
    lobby_opponent_policy: str,
    allow_oracle_macro_actions: bool,
) -> EmulatedTrainingEnvBatch | NativePufferTrainingEnvBatch:
    if puffer_backend == "native":
        if env_kind == "lobby":
            if not isinstance(config, EnvConfig):
                raise TypeError("native lobby env requires EnvConfig")
            return NativePufferTrainingEnvBatch(
                make_lobby_hero_native_puffer_env(
                    num_envs=num_envs,
                    seed=seed,
                    config=config,
                    player_count=players,
                    opponent_policy=LOBBY_POLICY_BY_NAME[lobby_opponent_policy],
                    max_actions_per_player=max_actions_per_player,
                    allow_oracle_macro_actions=allow_oracle_macro_actions,
                )
            )
        if env_kind == "strategic":
            if not isinstance(config, StrategicConfig):
                raise TypeError("native strategic env requires StrategicConfig")
            return NativePufferTrainingEnvBatch(
                make_strategic_native_puffer_env(
                    num_envs=num_envs,
                    seed=seed,
                    config=config,
                )
            )
        raise ValueError("--puffer-backend native supports --env-kind lobby or strategic")
    if puffer_backend == "emulated":
        return EmulatedTrainingEnvBatch(
            [
                make_training_env(
                    env_kind=env_kind,
                    seed=seed + env_index,
                    config=config,
                    players=players,
                    max_actions_per_player=max_actions_per_player,
                    lobby_opponent_policy=lobby_opponent_policy,
                    allow_oracle_macro_actions=allow_oracle_macro_actions,
                )
                for env_index in range(num_envs)
            ]
        )
    raise ValueError(f"unsupported Puffer backend: {puffer_backend}")


def main() -> None:
    args = parse_args()
    train_puffer_ppo(args)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("checkpoints/puffer_ppo.pt"))
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--n-steps", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--update-epochs", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2.5e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-range", type=float, default=0.2)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--env-kind",
        choices=["single_player", "lobby", "strategic"],
        default="lobby",
    )
    parser.add_argument(
        "--puffer-backend",
        choices=["emulated", "native"],
        default="emulated",
        help="Use current Gymnasium emulation or native batched PufferEnv for lobby PPO.",
    )
    parser.add_argument(
        "--lobby-opponent-policy",
        choices=sorted(LOBBY_POLICY_BY_NAME),
        default="tempo",
    )
    parser.add_argument("--players", type=int, default=8)
    parser.add_argument("--max-round", type=int, default=36)
    parser.add_argument("--max-actions-per-round", type=int, default=20)
    parser.add_argument("--max-actions-per-player", type=int, default=None)
    parser.add_argument(
        "--disallow-oracle-macro-actions",
        action="store_true",
        help="For lobby PPO, remove field_best_board and slam_best_item from player-0 masks.",
    )
    parser.add_argument(
        "--hidden-sizes",
        default="64,64",
        help="Comma-separated MLP hidden sizes.",
    )
    return parser.parse_args(argv)


def train_puffer_ppo(args: argparse.Namespace) -> dict[str, Any]:
    if args.num_envs < 1:
        raise ValueError("--num-envs must be at least 1")
    if args.n_steps < 1:
        raise ValueError("--n-steps must be at least 1")

    device = resolve_device(args.device)
    set_seed(args.seed)
    env_kind = cast(PufferEnvKind, args.env_kind)
    config: TrainingConfig
    if env_kind == "strategic":
        config = StrategicConfig(
            max_round=args.max_round,
            max_actions_per_round=args.max_actions_per_round,
        )
    else:
        config = EnvConfig(
            seed=args.seed,
            max_round=args.max_round,
            max_actions_per_round=args.max_actions_per_round,
        )
    puffer_backend = cast(PufferBackend, args.puffer_backend)
    env_batch = make_training_env_batch(
        env_kind=env_kind,
        puffer_backend=puffer_backend,
        seed=args.seed,
        config=config,
        num_envs=args.num_envs,
        players=args.players,
        max_actions_per_player=args.max_actions_per_player,
        lobby_opponent_policy=args.lobby_opponent_policy,
        allow_oracle_macro_actions=not args.disallow_oracle_macro_actions,
    )
    started = time.perf_counter()
    try:
        observations, reset_infos = env_batch.reset(args.seed)
        action_space = cast(spaces.Discrete, env_batch.single_action_space)
        action_dim = int(action_space.n)
        hidden_sizes = parse_hidden_sizes(args.hidden_sizes)
        policy = MaskedActorCritic(
            observation_dim=observations.shape[1],
            action_dim=action_dim,
            hidden_sizes=hidden_sizes,
        ).to(device)
        optimizer = torch.optim.Adam(policy.parameters(), lr=args.learning_rate, eps=1e-5)
        rollout_size, batch_size = resolve_rollout_and_batch_size(args)
        summary = run_training_loop(
            env_batch=env_batch,
            policy=policy,
            optimizer=optimizer,
            observations=observations,
            args=args,
            device=device,
            action_dim=action_dim,
            rollout_size=rollout_size,
            batch_size=batch_size,
        )
        checkpoint_path = puffer_checkpoint_path(args.out)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "kind": "puffer_ppo",
                "model_state_dict": policy.state_dict(),
                "args": json_ready(vars(args)),
                "resolved": {
                    "action_dim": action_dim,
                    "observation_dim": int(observations.shape[1]),
                    "rollout_size": rollout_size,
                    "batch_size": batch_size,
                    "device": str(device),
                    "hidden_sizes": hidden_sizes,
                    "puffer_backend": puffer_backend,
                },
                "summary": summary,
            },
            checkpoint_path,
        )
        manifest_path = write_puffer_manifest(
            output=checkpoint_path,
            args=args,
            reset_infos=reset_infos,
            resolved={
                "action_dim": action_dim,
                "observation_dim": int(observations.shape[1]),
                "rollout_size": rollout_size,
                "batch_size": batch_size,
                "device": str(device),
                "hidden_sizes": hidden_sizes,
                "puffer_backend": puffer_backend,
            },
            summary=summary,
            elapsed_sec=time.perf_counter() - started,
        )
        print_resolved_config(args, rollout_size=rollout_size, batch_size=batch_size)
        print(f"checkpoint: `{checkpoint_path}`")
        print(f"manifest: `{manifest_path}`")
        return {
            "checkpoint_path": str(checkpoint_path),
            "manifest_path": str(manifest_path),
            "summary": summary,
        }
    finally:
        env_batch.close()


def run_training_loop(
    *,
    env_batch: EmulatedTrainingEnvBatch | NativePufferTrainingEnvBatch,
    policy: MaskedActorCritic,
    optimizer: torch.optim.Optimizer,
    observations: NDArray[np.float32],
    args: argparse.Namespace,
    device: torch.device,
    action_dim: int,
    rollout_size: int,
    batch_size: int,
) -> dict[str, Any]:
    num_envs = env_batch.num_envs
    obs_dim = int(observations.shape[1])
    num_updates = max(1, int(np.ceil(args.timesteps / rollout_size)))
    next_obs = observations.astype(np.float32)
    episode_returns = np.zeros(num_envs, dtype=np.float32)
    episode_lengths = np.zeros(num_envs, dtype=np.int32)
    completed_returns: list[float] = []
    completed_lengths: list[int] = []
    next_reset_seeds = np.arange(args.seed + 10_000, args.seed + 10_000 + num_envs)

    for _update in range(num_updates):
        obs_buf = np.zeros((args.n_steps, num_envs, obs_dim), dtype=np.float32)
        actions_buf = np.zeros((args.n_steps, num_envs), dtype=np.int64)
        logprobs_buf = np.zeros((args.n_steps, num_envs), dtype=np.float32)
        rewards_buf = np.zeros((args.n_steps, num_envs), dtype=np.float32)
        dones_buf = np.zeros((args.n_steps, num_envs), dtype=np.float32)
        values_buf = np.zeros((args.n_steps, num_envs), dtype=np.float32)

        for step in range(args.n_steps):
            obs_buf[step] = next_obs
            obs_tensor = torch.as_tensor(next_obs, dtype=torch.float32, device=device)
            with torch.no_grad():
                actions, logprobs, _entropy, values = policy.get_action_and_value(obs_tensor)
            actions_np = actions.cpu().numpy().astype(np.int64)
            actions_buf[step] = actions_np
            logprobs_buf[step] = logprobs.cpu().numpy()
            values_buf[step] = values.cpu().numpy()

            step_obs, rewards, terminals, truncations, _infos = env_batch.step(actions_np)
            dones = np.logical_or(terminals, truncations)
            for env_index, done_value in enumerate(dones):
                done = bool(done_value)
                reward = float(rewards[env_index])
                rewards_buf[step, env_index] = reward
                dones_buf[step, env_index] = float(done)
                episode_returns[env_index] += reward
                episode_lengths[env_index] += 1
                if done:
                    completed_returns.append(float(episode_returns[env_index]))
                    completed_lengths.append(int(episode_lengths[env_index]))
                    episode_returns[env_index] = 0.0
                    episode_lengths[env_index] = 0
                    if not env_batch.auto_resets:
                        reset_obs, _ = env_batch.reset_one(
                            env_index,
                            seed=int(next_reset_seeds[env_index]),
                        )
                        step_obs[env_index] = reset_obs
                    next_reset_seeds[env_index] += num_envs
                next_obs[env_index] = np.asarray(step_obs[env_index], dtype=np.float32).reshape(-1)

        with torch.no_grad():
            next_obs_tensor = torch.as_tensor(next_obs, dtype=torch.float32, device=device)
            _actions, _logprobs, _entropy, next_values = policy.get_action_and_value(
                next_obs_tensor
            )
        advantages = compute_gae(
            rewards=rewards_buf,
            dones=dones_buf,
            values=values_buf,
            next_values=next_values.cpu().numpy(),
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
        )
        returns = advantages + values_buf

        train_minibatches(
            policy=policy,
            optimizer=optimizer,
            obs=obs_buf.reshape(rollout_size, obs_dim),
            actions=actions_buf.reshape(rollout_size),
            old_logprobs=logprobs_buf.reshape(rollout_size),
            advantages=advantages.reshape(rollout_size),
            returns=returns.reshape(rollout_size),
            args=args,
            device=device,
            batch_size=batch_size,
            action_dim=action_dim,
        )

    total_steps = num_updates * rollout_size
    return {
        "updates": num_updates,
        "total_steps": total_steps,
        "completed_episodes": len(completed_returns),
        "mean_episode_return": float(np.mean(completed_returns)) if completed_returns else None,
        "mean_episode_length": float(np.mean(completed_lengths)) if completed_lengths else None,
    }


def compute_gae(
    *,
    rewards: NDArray[np.float32],
    dones: NDArray[np.float32],
    values: NDArray[np.float32],
    next_values: NDArray[np.float32],
    gamma: float,
    gae_lambda: float,
) -> NDArray[np.float32]:
    advantages = np.zeros_like(rewards, dtype=np.float32)
    last_gae = np.zeros(rewards.shape[1], dtype=np.float32)
    for step in reversed(range(rewards.shape[0])):
        if step == rewards.shape[0] - 1:
            next_value = next_values
        else:
            next_value = values[step + 1]
        next_nonterminal = 1.0 - dones[step]
        delta = rewards[step] + gamma * next_value * next_nonterminal - values[step]
        last_gae = delta + gamma * gae_lambda * next_nonterminal * last_gae
        advantages[step] = last_gae
    return advantages


def train_minibatches(
    *,
    policy: MaskedActorCritic,
    optimizer: torch.optim.Optimizer,
    obs: NDArray[np.float32],
    actions: NDArray[np.int64],
    old_logprobs: NDArray[np.float32],
    advantages: NDArray[np.float32],
    returns: NDArray[np.float32],
    args: argparse.Namespace,
    device: torch.device,
    batch_size: int,
    action_dim: int,
) -> None:
    del action_dim
    obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device)
    actions_tensor = torch.as_tensor(actions, dtype=torch.int64, device=device)
    old_logprobs_tensor = torch.as_tensor(old_logprobs, dtype=torch.float32, device=device)
    advantages_tensor = torch.as_tensor(advantages, dtype=torch.float32, device=device)
    returns_tensor = torch.as_tensor(returns, dtype=torch.float32, device=device)
    advantages_tensor = (advantages_tensor - advantages_tensor.mean()) / (
        advantages_tensor.std(unbiased=False) + 1e-8
    )
    sample_count = obs_tensor.shape[0]

    for _epoch in range(args.update_epochs):
        indices = torch.randperm(sample_count, device=device)
        for start in range(0, sample_count, batch_size):
            minibatch = indices[start : start + batch_size]
            _action, new_logprob, entropy, new_value = policy.get_action_and_value(
                obs_tensor[minibatch],
                actions_tensor[minibatch],
            )
            logratio = new_logprob - old_logprobs_tensor[minibatch]
            ratio = logratio.exp()
            minibatch_advantages = advantages_tensor[minibatch]
            policy_loss_1 = -minibatch_advantages * ratio
            policy_loss_2 = -minibatch_advantages * torch.clamp(
                ratio,
                1.0 - args.clip_range,
                1.0 + args.clip_range,
            )
            policy_loss = torch.max(policy_loss_1, policy_loss_2).mean()
            value_loss = 0.5 * (new_value - returns_tensor[minibatch]).pow(2).mean()
            entropy_loss = entropy.mean()
            loss = policy_loss - args.ent_coef * entropy_loss + args.vf_coef * value_loss

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), args.max_grad_norm)
            optimizer.step()


def reset_envs(envs: list[Any], seed: int) -> tuple[NDArray[np.float32], list[dict[str, Any]]]:
    observations: list[NDArray[np.float32]] = []
    infos: list[dict[str, Any]] = []
    for env_index, env in enumerate(envs):
        obs, info = env.reset(seed=seed + env_index)
        observations.append(np.asarray(obs, dtype=np.float32).reshape(-1))
        infos.append(info)
    return np.stack(observations).astype(np.float32), infos


def resolve_rollout_and_batch_size(args: argparse.Namespace) -> tuple[int, int]:
    rollout_size = int(args.n_steps * args.num_envs)
    batch_size = args.batch_size if args.batch_size is not None else min(4096, rollout_size)
    return rollout_size, max(1, min(int(batch_size), rollout_size))


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def puffer_checkpoint_path(output: Path) -> Path:
    if output.suffix == ".pt":
        return output
    return output.with_suffix(".pt")


def print_resolved_config(args: argparse.Namespace, *, rollout_size: int, batch_size: int) -> None:
    print("# Puffer PPO Training")
    print(f"- env_kind: {args.env_kind}")
    print(f"- puffer_backend: {args.puffer_backend}")
    print(f"- lobby_opponent_policy: {args.lobby_opponent_policy}")
    print(f"- num_envs: {args.num_envs}")
    print(f"- n_steps: {args.n_steps}")
    print(f"- rollout_size: {rollout_size}")
    print(f"- batch_size: {batch_size}")
    print(f"- timesteps_requested: {args.timesteps}")


def write_puffer_manifest(
    *,
    output: Path,
    args: argparse.Namespace,
    reset_infos: list[dict[str, Any]],
    resolved: dict[str, Any],
    summary: dict[str, Any],
    elapsed_sec: float,
) -> Path:
    manifest_path = output.with_suffix(".manifest.json")
    manifest = {
        "kind": "puffer_ppo",
        "created_at_unix": time.time(),
        "elapsed_sec": elapsed_sec,
        "model_path": str(output),
        "args": json_ready(vars(args)),
        "resolved": json_ready(resolved),
        "summary": json_ready(summary),
        "initial_reset_info_keys": [sorted(info.keys()) for info in reset_infos],
        "git": git_metadata(),
        "packages": package_versions(),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest_path


def json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [json_ready(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def git_metadata() -> dict[str, Any]:
    status = git_output("status", "--porcelain=v1")
    return {
        "commit": git_output("rev-parse", "HEAD"),
        "branch": git_output("rev-parse", "--abbrev-ref", "HEAD"),
        "status_porcelain": status,
        "dirty": bool(status),
    }


def git_output(*args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args],
            check=False,
            capture_output=True,
            text=True,
            timeout=2,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def package_versions() -> dict[str, str]:
    packages = ["tft-zero", "gymnasium", "numpy", "pufferlib", "torch"]
    versions: dict[str, str] = {}
    for package in packages:
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            continue
    return versions


if __name__ == "__main__":
    main()

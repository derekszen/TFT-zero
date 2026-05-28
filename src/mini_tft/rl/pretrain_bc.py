"""Behavioral cloning warm start for MaskablePPO policies."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from mini_tft.rl.dataset import RolloutDataset, load_dataset
from mini_tft.rl.train_ppo import make_env


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("checkpoints/bc_v0"))
    parser.add_argument("--init", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--n-steps", type=int, default=256)
    parser.add_argument("--ppo-batch-size", type=int, default=2048)
    parser.add_argument("--entropy-coef", type=float, default=0.001)
    parser.add_argument(
        "--hidden-sizes",
        default="64,64",
        help="Comma-separated MLP hidden sizes used for new policies.",
    )
    args = parser.parse_args()

    try:
        import torch
        from sb3_contrib import MaskablePPO
        from stable_baselines3.common.vec_env import DummyVecEnv
    except ImportError as exc:
        raise SystemExit("Install training dependencies with `uv sync --extra train`.") from exc

    dataset = load_dataset(args.dataset)
    train_indices, val_indices = split_indices(len(dataset.actions), args.val_fraction, args.seed)
    env = DummyVecEnv([make_env(args.seed)])

    try:
        if args.init is None:
            model = MaskablePPO(
                "MlpPolicy",
                env,
                n_steps=args.n_steps,
                batch_size=args.ppo_batch_size,
                learning_rate=args.learning_rate,
                gamma=0.99,
                gae_lambda=0.95,
                clip_range=0.2,
                ent_coef=0.01,
                vf_coef=0.5,
                policy_kwargs={"net_arch": parse_hidden_sizes(args.hidden_sizes)},
                verbose=0,
                seed=args.seed,
                device=args.device,
            )
        else:
            model = MaskablePPO.load(args.init, env=env, device=args.device)

        optimizer = torch.optim.Adam(model.policy.parameters(), lr=args.learning_rate)
        rng = np.random.default_rng(args.seed)
        started = time.perf_counter()

        print("# Behavioral Cloning Pretrain")
        print(f"- dataset: `{args.dataset}`")
        print(f"- train transitions: {len(train_indices)}")
        print(f"- validation transitions: {len(val_indices)}")
        print()
        print("| epoch | train_loss | train_acc | val_acc | elapsed_sec |")
        print("| ---: | ---: | ---: | ---: | ---: |")

        for epoch in range(1, args.epochs + 1):
            rng.shuffle(train_indices)
            loss_total = 0.0
            correct = 0
            seen = 0
            for batch_indices in batches(train_indices, args.batch_size):
                obs, actions, masks = batch_tensors(dataset, batch_indices, model.policy.device)
                optimizer.zero_grad(set_to_none=True)
                values, log_prob, entropy = model.policy.evaluate_actions(
                    obs,
                    actions,
                    action_masks=masks,
                )
                del values
                entropy_loss = -entropy.mean() if entropy is not None else 0.0
                loss = -log_prob.mean() + args.entropy_coef * entropy_loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.policy.parameters(), 0.5)
                optimizer.step()

                loss_total += float(loss.item()) * len(batch_indices)
                with torch.no_grad():
                    distribution = model.policy.get_distribution(obs, action_masks=masks)
                    logits = distribution.distribution.logits
                    predicted = logits.argmax(dim=1)
                    correct += int((predicted == actions).sum().item())
                seen += len(batch_indices)

            val_acc = masked_argmax_accuracy(model, dataset, val_indices, args.batch_size)
            elapsed = time.perf_counter() - started
            print(
                f"| {epoch} | {loss_total / max(seen, 1):.4f} | "
                f"{correct / max(seen, 1):.4f} | {val_acc:.4f} | {elapsed:.1f} |"
            )

        args.out.parent.mkdir(parents=True, exist_ok=True)
        model.save(args.out)
        print()
        print(f"saved: `{args.out}.zip`")
    finally:
        env.close()


def split_indices(n: int, val_fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    if not 0 <= val_fraction < 1:
        raise ValueError("--val-fraction must be in [0, 1)")
    indices = np.arange(n, dtype=np.int64)
    np.random.default_rng(seed).shuffle(indices)
    val_count = int(n * val_fraction)
    return indices[val_count:], indices[:val_count]


def batches(indices: np.ndarray, batch_size: int) -> list[np.ndarray]:
    if batch_size < 1:
        raise ValueError("--batch-size must be positive")
    return [indices[start : start + batch_size] for start in range(0, len(indices), batch_size)]


def batch_tensors(dataset: RolloutDataset, indices: np.ndarray, device: object):
    import torch

    obs = torch.as_tensor(dataset.obs[indices], dtype=torch.float32, device=device)
    actions = torch.as_tensor(dataset.actions[indices], dtype=torch.long, device=device)
    masks = torch.as_tensor(dataset.masks[indices], dtype=torch.bool, device=device)
    return obs, actions, masks


def masked_argmax_accuracy(
    model: object,
    dataset: RolloutDataset,
    indices: np.ndarray,
    batch_size: int,
) -> float:
    import torch

    if len(indices) == 0:
        return float("nan")

    correct = 0
    seen = 0
    with torch.no_grad():
        for batch_indices in batches(indices, batch_size):
            obs, actions, masks = batch_tensors(dataset, batch_indices, model.policy.device)
            distribution = model.policy.get_distribution(obs, action_masks=masks)
            logits = distribution.distribution.logits
            predicted = logits.argmax(dim=1)
            correct += int((predicted == actions).sum().item())
            seen += len(batch_indices)
    return correct / seen


def parse_hidden_sizes(value: str) -> list[int]:
    sizes = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not sizes:
        raise ValueError("--hidden-sizes must include at least one integer")
    if any(size <= 0 for size in sizes):
        raise ValueError("--hidden-sizes must contain positive integers")
    return sizes


if __name__ == "__main__":
    main()

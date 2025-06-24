from __future__ import annotations
import argparse
import csv
import os
from pathlib import Path
from typing import List, Dict, Tuple

import gym
import numpy as np
import torch
from stable_baselines3 import PPO
from env.custom_hopper import *      

# Define project directories
BASE_DIR      = Path(__file__).resolve().parents[2]  # project root
WEIGHTS_DIR   = BASE_DIR / "models_weights"
DATA_DIR      = BASE_DIR / "models_data"
SOURCE_ENV_ID = "CustomHopper-source-v0"
TARGET_ENV_ID = "CustomHopper-target-v0"

def _evaluate_single(
    model_path: Path,
    env_id: str,
    episodes: int,
    seed: int,
    device: str,
    render: bool
) -> Tuple[float, float]:
    """Run one model checkpoint on a single environment and return (mean, std) reward."""
    env = gym.make(env_id)
    env.seed(seed)
    env.action_space.seed(seed)

    model: PPO = PPO.load(str(model_path), device=device)

    rewards: List[float] = []
    for _ in range(episodes):
        obs, done, total_r = env.reset(), False, 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, r, done, _ = env.step(action)
            total_r += float(r)
            if render:
                env.render()
        rewards.append(total_r)

    env.close()
    return float(np.mean(rewards)), float(np.std(rewards))

def evaluate_all_ppo(
    train_env: str,
    test_envs: List[str],
    seed: int,
    episodes: int,
    device: str = "cpu",
    render: bool = False,
    use_udr_flag: bool = False
) -> List[Dict[str, float]]:
    """
    Evaluate a specific training environment against one or more test envs.
    If --use-udr flag is not set, will try both UDR disabled and enabled.
    """
    results: List[Dict[str, float]] = []
    udr_options = (True,) if use_udr_flag else (False, True)

    for use_udr in udr_options:
        ckpt = WEIGHTS_DIR / f"ppo_{train_env}_seed_{seed}_UDR_{use_udr}.zip"
        if not ckpt.exists():
            continue

        for t in test_envs:
            env_id = SOURCE_ENV_ID if t == "source" else TARGET_ENV_ID
            mean_r, std_r = _evaluate_single(
                model_path=ckpt,
                env_id=env_id,
                episodes=episodes,
                seed=seed,
                device=device,
                render=render
            )
            print(f"{train_env}→{t} | UDR={use_udr} | Return {mean_r:.2f} ± {std_r:.2f}")
            results.append({
                "train_env": train_env,
                "test_env": t,
                "udr": use_udr,
                "mean_return": mean_r,
                "std_return": std_r
            })

    # Save the results to CSV
    if results:
        DATA_DIR.mkdir(exist_ok=True)
        csv_path = DATA_DIR / f"PPO_test_{train_env}_seed_{seed}_ep{episodes}.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["train_env", "test_env", "udr", "mean_return", "std_return"]
            )
            writer.writeheader()
            writer.writerows(results)
        print(f"\nSaved evaluation log ➜ {csv_path}")

    return results


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate PPO checkpoints on CustomHopper environments"
    )
    parser.add_argument(
        "--env", required=True, choices=["source", "target"],
        help="Training environment: source or target"
    )
    parser.add_argument(
        "--test-env", choices=["source", "target"], required=False,
        help="Optional filter: only test this environment"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--episodes", type=int, default=100_000,
        help="Number of episodes to evaluate"
    )
    parser.add_argument(
        "--device", choices=["cpu", "cuda"], default="cpu",
        help="Device to run inference on"
    )
    parser.add_argument(
        "--render", action="store_true",
        help="Render the environment during evaluation"
    )
    parser.add_argument(
        "--use-udr", action="store_true",
        help="If set, only evaluate UDR-enabled checkpoints"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Determine which test environments to run
    if args.test_env:
        test_envs = [args.test_env]
    else:
        test_envs = ["source", "target"]

    # Run evaluation for the specified training env and test envs
    evaluate_all_ppo(
        train_env=args.env,
        test_envs=test_envs,
        seed=args.seed,
        episodes=args.episodes,
        device=args.device,
        render=args.render,
        use_udr_flag=args.use_udr
    )




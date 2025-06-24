from __future__ import annotations
import csv
import os
from pathlib import Path
from typing import List, Dict, Tuple

import gym
import numpy as np
import torch
from stable_baselines3 import PPO
from env.custom_hopper import *      

BASE_DIR       = Path(__file__).resolve().parents[2]      # project root
WEIGHTS_DIR    = BASE_DIR / "models_weights"
DATA_DIR       = BASE_DIR / "models_data"
SOURCE_ENV_ID  = "CustomHopper-source-v0"
TARGET_ENV_ID  = "CustomHopper-target-v0"


def _evaluate_single(
    model_path: Path,
    env_id: str,
    episodes: int,
    seed: int,
    device: str,
    render: bool
):
    """Return (mean, std) return over `episodes` episodes."""
    env = gym.make(env_id)
    env.seed(seed)
    env.action_space.seed(seed)

    model: PPO = PPO.load(str(model_path), device=device)

    returns: List[float] = []
    for _ in range(episodes):
        obs, done, tot_r = env.reset(), False, 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _ = env.step(action)
            tot_r += float(reward)
            if render:
                env.render()
        returns.append(tot_r)

    env.close()
    return float(np.mean(returns)), float(np.std(returns))


def evaluate_all_ppo(
    seed: int,
    episodes: int,
    device: str = "cpu",
    render: bool = False,
):
    """
    Evaluate every (source/target × UDR True/False) checkpoint and
    return a list of results.  Also writes the CSV described above.
    """
    results: List[Dict[str, str | float]] = []

    for env_tag in ("source", "target"):
        for use_udr in (False, True):
            ckpt = WEIGHTS_DIR / f"ppo_{env_tag}_seed_{seed}_UDR_{use_udr}.zip"
            if not ckpt.exists():
                # silently skip missing combinations
                continue

            for test_env in (SOURCE_ENV_ID, TARGET_ENV_ID):
                label = f"{env_tag}→{'source' if test_env==SOURCE_ENV_ID else 'target'}"
                mean_r, std_r = _evaluate_single(
                    model_path = ckpt,
                    env_id     = test_env,
                    episodes   = episodes,
                    seed       = seed,
                    device     = device,
                    render     = render
                )
                results.append({
                    "checkpoint": ckpt.name,
                    "train_env": env_tag,
                    "udr": use_udr,
                    "test_env": "source" if test_env == SOURCE_ENV_ID else "target",
                    "mean_return": mean_r,
                    "std_return": std_r,
                })
                print(f"{label:<13} | UDR {use_udr:<5} | "
                      f"Return {mean_r:8.2f} ± {std_r:.2f}")


    # save 
    DATA_DIR.mkdir(exist_ok=True)
    csv_name = f"PPO_test_seed_{seed}_ep{episodes}.csv"
    csv_path = DATA_DIR / csv_name
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "checkpoint", "train_env", "udr",
                "test_env", "mean_return", "std_return"
            ]
        )
        writer.writeheader()
        writer.writerows(results)

    print(f"\nSaved evaluation log ➜ {csv_path}")
    return results



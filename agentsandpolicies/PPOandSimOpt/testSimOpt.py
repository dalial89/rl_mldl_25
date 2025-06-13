"""
testSimOpt.py
--------------
Evaluation script for SimOpt models (plus vanilla/UDR). Can be invoked via
main.py or standalone.
"""
import argparse
import random
import sys
from pathlib import Path
from typing import List, Tuple

import gym
import numpy as np
import pandas as pd
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

# ----------------------- import project root --------------------------------
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from env.custom_hopper import *  # noqa: F401,F403 – register envs

# ---------------------------------------------------------------------------
# utilities
# ---------------------------------------------------------------------------

def set_seeds(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_eval_env(env_id: str, seed: int):
    env = gym.make(env_id)
    env.seed(seed)
    return DummyVecEnv([lambda: Monitor(env)])


# ---------------------------------------------------------------------------
# one-shot evaluator
# ---------------------------------------------------------------------------

def evaluate_model(model_path: Path, env_id: str, logs_dir: Path,
                   episodes: int, render: bool, device: str, seed: int) -> Tuple[float, float]:
    vec_env = make_eval_env(env_id, seed)

    norm_path = logs_dir / "vecnormalize.pkl"
    if norm_path.exists():
        vec_env = VecNormalize.load(str(norm_path), vec_env)
        vec_env.training = False; vec_env.norm_reward = False

    model = PPO.load(str(model_path), env=vec_env, device=device)

    returns: List[float] = []
    for ep in range(episodes):
        obs, done, ep_ret = vec_env.reset(), False, 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _ = vec_env.step(action)
            ep_ret += float(reward)
            if render:
                vec_env.render()
        returns.append(ep_ret)
    vec_env.close()
    return float(np.mean(returns)), float(np.std(returns))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--episodes", type=int, default=50)
    p.add_argument("--device", default="cpu")
    p.add_argument("--render", action="store_true")
    p.add_argument("--udr", action="store_true")
    p.add_argument("--no-simopt", action="store_true")
    args = p.parse_args()

    set_seeds(args.seed)

    WEIGHTS = ROOT / "models_weights"
    LOGS    = Path(__file__).resolve().parent
    OUT     = ROOT / "training_data"; OUT.mkdir(exist_ok=True)

    MODEL = {
        "source_base": f"source_base_seed{args.seed}_final.zip",
        "source_udr":  f"source_udr_seed{args.seed}_final.zip",
        "target_base": f"target_base_seed{args.seed}_final.zip",
        "simopt":      f"simopt_final_seed{args.seed}.zip",
    }
    LOG = {
        key: f"{val.replace('.zip','').replace('final','logs')}" for key, val in MODEL.items()
    }

    tests = [
        ("source→source", "source_base", "CustomHopper-source-v0"),
        ("source→target", "source_base", "CustomHopper-target-v0"),
        ("target→target", "target_base", "CustomHopper-target-v0"),
    ]
    if args.udr:
        tests.insert(0, ("source(UDR)→source", "source_udr", "CustomHopper-source-v0"))
        tests.insert(1, ("source(UDR)→target", "source_udr", "CustomHopper-target-v0"))
    if not args.no_simopt:
        tests.extend([
            ("simopt→source", "simopt", "CustomHopper-source-v0"),
            ("simopt→target", "simopt", "CustomHopper-target-v0"),
        ])

    rows = []
    for label, tag, env_id in tests:
        mfile = WEIGHTS / MODEL[tag]
        ldir  = LOGS / LOG[tag]
        if not mfile.exists():
            print(f"[skip] {label}: model not found {mfile}")
            continue
        print(f"\n--- {label} ({env_id}) ---")
        mean, std = evaluate_model(mfile, env_id, ldir, args.episodes,
                                   args.render, args.device, args.seed)
        print(f" → {mean:.2f} ± {std:.2f}")
        rows.append(dict(setup=label, env=env_id, mean=mean, std=std,
                         seed=args.seed, episodes=args.episodes,
                         udr=args.udr, simopt=not args.no_simopt))

    if rows:
        df_path = OUT / f"testSimOpt_seed{args.seed}.csv"
        pd.DataFrame(rows).to_csv(df_path, index=False)
        print(f"\nSaved results to {df_path}")
    else:
        print("No models evaluated – nothing to save.")


if __name__ == "__main__":
    main()
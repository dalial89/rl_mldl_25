"""
trainSimOpt.py
--------------
*One‑stop* trainer that now produces **all** the artefacts expected by the
test script:

* `source_base_seed<seed>_final.zip`   – vanilla PPO on source
* `target_base_seed<seed>_final.zip`   – vanilla PPO on target
* `simopt_final_seed<seed>.zip`        – SimOpt‑optimised PPO (source env)

Run via:
```bash
python trainSimOpt.py --seed 42 --device cpu
# oppure
python main.py --simopt_train --seed 42 --device cuda:0
```
"""
import argparse
import random
from pathlib import Path
from typing import Optional

import gym
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from skopt import gp_minimize
from skopt.space import Real

from env.custom_hopper import *  # noqa: F401,F403 – register envs
from utils_simopt import (
    HopperMassRandomGaussianWrapper,
    gap as gap_fn,
    get_obs,
)

# ---------------------------------------------------------------------------
# constants
# ---------------------------------------------------------------------------
SEED = 42
SOURCE_ENV = "CustomHopper-source-v0"
TARGET_ENV = "CustomHopper-target-v0"

# vanilla PPO
TOTAL_TIMESTEPS_BASE = 2_000

# simopt
MAX_STEPS       = 5
BO_CALLS        = 10
TIMESTEPS_BO    = 100_000
TIMESTEPS_FINAL = 2_000_000
BETA            = 0.1

MODEL_DIR = Path("models_weights"); MODEL_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# utils
# ---------------------------------------------------------------------------

def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_env(env_id: str, seed: int, phi: Optional[dict] = None):
    env = gym.make(env_id)
    if phi is not None:
        env = HopperMassRandomGaussianWrapper(env, phi)
    env.seed(seed); env.action_space.seed(seed)
    return Monitor(env)


def train_vanilla(env_id: str, tag: str, device: str):
    """Train plain PPO and save to models_weights/<tag>_seed<seed>_final.zip"""
    out_path = MODEL_DIR / f"{tag}_seed{SEED}_final.zip"
    if out_path.exists():
        print(f"[skip] {out_path.name} already exists")
        return
    print(f"\n=== Vanilla PPO on {env_id} ({tag}) ===")
    env = make_env(env_id, SEED)
    model = PPO("MlpPolicy", env, seed=SEED, verbose=0, device=device)
    model.learn(total_timesteps=TOTAL_TIMESTEPS_BASE)
    model.save(str(out_path)); env.close()
    mean, std = evaluate_policy(model, make_env(env_id, SEED), n_eval_episodes=20, deterministic=True)
    print(f"Return: {mean:.1f} ± {std:.1f}")


def train_simopt(device: str):
    """Adaptive Gaussian SimOpt → returns path to final model."""
    phi = {"thigh": [3.93, 0.5], "leg": [2.71, 0.5], "foot": [5.09, 0.5]}
    env0 = make_env(SOURCE_ENV, SEED, phi)
    model = PPO("MlpPolicy", env0, seed=SEED, verbose=0, device=device)

    def bo_obj(x):
        # build phi candidate
        phi_cand = {k: [x[i], phi[k][1]] for i, k in enumerate(phi)}
        # short fine‑tune
        model.set_env(make_env(SOURCE_ENV, SEED, phi_cand))
        model.learn(total_timesteps=TIMESTEPS_BO, reset_num_timesteps=False)
        # gap + return
        real = get_obs(model, make_env(TARGET_ENV, SEED), n_episodes=3)
        sim  = get_obs(model, make_env(SOURCE_ENV, SEED, phi_cand), n_episodes=3)
        L = min(min(len(t) for t in real), min(len(t) for t in sim))
        gap = gap_fn([t[:L] for t in real], [t[:L] for t in sim])
        ret, _ = evaluate_policy(model, make_env(TARGET_ENV, SEED), n_eval_episodes=3, deterministic=True)
        return gap - BETA * ret

    for step in range(MAX_STEPS):
        print(f"\n--- SimOpt step {step} ---")
        space = [Real(m-2*s, m+2*s, name=k) for k, (m, s) in phi.items()]
        res = gp_minimize(bo_obj, space, n_calls=BO_CALLS, random_state=SEED+step)
        best = res.x; print("Best μ:", best, "obj", res.fun)
        # EMA update
        for i, k in enumerate(phi):
            mu, sig = phi[k]
            phi[k] = [0.7*mu + 0.3*best[i], max(0.7*sig, 1e-3)]

    final_path = MODEL_DIR / f"simopt_final_seed{SEED}.zip"
    train_env = make_env(SOURCE_ENV, SEED, phi)
    final_model = PPO("MlpPolicy", train_env, seed=SEED, verbose=0, device=device)
    final_model.learn(total_timesteps=TIMESTEPS_FINAL)
    final_model.save(str(final_path)); train_env.close()
    mean, std = evaluate_policy(final_model, make_env(SOURCE_ENV, SEED, phi), 20, True)
    print(f"SimOpt final return: {mean:.1f} ± {std:.1f}")
    return final_path

# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# entry‑point for main.py AND stand‑alone CLI
# ---------------------------------------------------------------------------

def run_simopt(*, seed: int = 42, device: str = "cpu"):
    """Callable from main.py to launch the full training pipeline."""
    _main(seed, device)


def _main(seed: int, device: str):
    """Internal main so we can reuse it for CLI and run_simopt."""
    global SEED
    SEED = seed
    set_seed(SEED)

    # 1) vanilla models
    train_vanilla(SOURCE_ENV, "source_base", device)
    train_vanilla(TARGET_ENV, "target_base", device)

    # 2) SimOpt model
    train_simopt(device)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    cli = argparse.ArgumentParser("Train PPO + SimOpt")
    cli.add_argument("--seed", type=int, default=42)
    cli.add_argument("--device", default="cpu")
    args = cli.parse_args()
    _main(args.seed, args.device)





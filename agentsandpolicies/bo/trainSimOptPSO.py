"""
trainSimOptPSO.py
-----------------
All‑in‑one trainer che produce:

1. **source_base_seed<seed>_final.zip**   – PPO vanilla su _source_
2. **target_base_seed<seed>_final.zip**   – PPO vanilla su _target_
3. **simopt_final_seed<seed>.zip**        – PPO addestrato con SimOpt (PSO)

Esempio:
```
python trainSimOptPSO.py --seed 42 --device cpu
```
"""

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Dict, List

import gym
import nevergrad as ng
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor

# -------------------------------------------------------------------------
#  IMPORTA e registra l'ambiente custom
#  (richiede che la cartella "env" sia sul PYTHONPATH e contenga __init__.py)
# -------------------------------------------------------------------------
import env.custom_hopper  # noqa: F401  - esegue la register dei 3 ID Gym

# -------------------------------------------------------------------------
#  Wrapper + funzioni di utilità – tutto in questo file (niente utils_simopt)
# -------------------------------------------------------------------------
class HopperMassRandomGaussianWrapper(gym.Wrapper):
    """Domain Randomization: campiona le prime tre masse da N(μ, σ²) a ogni reset."""

    _order = ["thigh", "leg", "foot"]  # mapping sulle prime 3 masse body_mass[1:]

    def __init__(self, env: gym.Env, phi: Dict[str, List[float]]):
        super().__init__(env)
        self.phi = phi

    def reset(self, **kwargs):
        masses = self.env.get_parameters().copy()
        for i, key in enumerate(self._order):
            mu, sig = self.phi[key]
            masses[i] = np.random.normal(mu, sig)
        self.env.set_parameters(masses)
        return self.env.reset(**kwargs)

    def step(self, action):
        return self.env.step(action)


def gap(real: List[np.ndarray], sim: List[np.ndarray]) -> float:
    """Simple L2 gap between flattened observation trajectories."""
    return float(np.linalg.norm(np.concatenate(real) - np.concatenate(sim)))


def get_obs(model: PPO, env: gym.Env, n_ep: int = 3) -> List[np.ndarray]:
    """Collects flattened observation trajectories over *n_ep* deterministic rollouts."""
    obs_trajs: List[np.ndarray] = []
    for _ in range(n_ep):
        done, traj = False, []
        obs = env.reset()
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, done, _ = env.step(action)
            traj.append(obs)
        obs_trajs.append(np.array(traj).flatten())
    env.close()
    return obs_trajs

# -------------------------------------------------------------------------
#  Costanti
# -------------------------------------------------------------------------
SOURCE_ENV = "CustomHopper-source-v0"
TARGET_ENV = "CustomHopper-target-v0"
TOTAL_STEPS_BASE = 2_000
SIMOPT_BUDGET = 40          # iterazioni PSO
SIGMA = 0.5
FINE_TUNE_STEPS = 40_000
FINAL_TRAIN_STEPS = 2_000_000
BETA = 0.1
MODEL_DIR = Path("models_weights"); MODEL_DIR.mkdir(exist_ok=True)

# -------------------------------------------------------------------------
#  Helper
# -------------------------------------------------------------------------

def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_env(env_id: str, seed: int, phi: Dict[str, List[float]] = None):
    env = gym.make(env_id)
    if phi is not None:
        env = HopperMassRandomGaussianWrapper(env, phi)
    env.seed(seed); env.action_space.seed(seed)
    return Monitor(env)


def train_vanilla(env_id: str, tag: str, seed: int, device: str):
    out = MODEL_DIR / f"{tag}_seed{seed}_final.zip"
    if out.exists():
        print(f"[skip] {out.name} esiste già"); return
    env = make_env(env_id, seed)
    model = PPO("MlpPolicy", env, seed=seed, verbose=0, device=device)
    model.learn(total_timesteps=TOTAL_STEPS_BASE)
    model.save(str(out)); env.close()
    m, s = evaluate_policy(model, make_env(env_id, seed), 20, True)
    print(f"{tag}: {m:.1f} ± {s:.1f}")


# -------------------------------------------------------------------------
#  SimOpt con PSO
# -------------------------------------------------------------------------

def train_simopt(seed: int, device: str):
    phi = {"thigh": [3.93, 0.5], "leg": [2.71, 0.5], "foot": [5.09, 0.5]}
    model = PPO("MlpPolicy", make_env(SOURCE_ENV, seed, phi), seed=seed, verbose=0, device=device)

    instr = ng.p.Dict(
        thigh=ng.p.Scalar(phi["thigh"][0]).set_bounds(0.1, 20).set_mutation(sigma=SIGMA),
        leg  =ng.p.Scalar(phi["leg"][0]).set_bounds(0.1, 20).set_mutation(sigma=SIGMA),
        foot =ng.p.Scalar(phi["foot"][0]).set_bounds(0.1, 20).set_mutation(sigma=SIGMA),
    )
    optim = ng.optimizers.PSO(parametrization=instr, budget=SIMOPT_BUDGET)

    def obj(mu_vals):
        # costruisci φ candidato con i μ proposti dal PSO
        phi_cand = {k: [mu_vals[k], phi[k][1]] for k in phi}
        model.set_env(make_env(SOURCE_ENV, seed, phi_cand))
        model.learn(total_timesteps=FINE_TUNE_STEPS, reset_num_timesteps=False)
        real = get_obs(model, make_env(TARGET_ENV, seed), 3)
        sim = get_obs(model, make_env(SOURCE_ENV, seed, phi_cand), 3)
        L = min(min(len(r) for r in real), min(len(s) for s in sim))
        g = gap([r[:L] for r in real], [s[:L] for s in sim])
        ret, _ = evaluate_policy(model, make_env(TARGET_ENV, seed), 3, True)
        return g - BETA * ret

    for k in range(SIMOPT_BUDGET):
        cand = optim.ask()
        loss = obj(cand.value)
        optim.tell(cand, loss)
        print(f"PSO iter {k}: loss {loss:.3f}")

    best_mu = optim.recommend().value
    for k in phi:
        phi[k][0] = best_mu[k]
    print("PSO – migliori μ:", best_mu)

    final_path = MODEL_DIR / f"simopt_final_seed{seed}.zip"
    train_env = make_env(SOURCE_ENV, seed, phi)
    final_model = PPO("MlpPolicy", train_env, seed=seed, verbose=0, device=device)
    final_model.learn(total_timesteps=FINAL_TRAIN_STEPS)
    final_model.save(str(final_path)); train_env.close()
    m, s = evaluate_policy(final_model, make_env(SOURCE_ENV, seed, phi), 20, True)
    print(f"SimOpt (PSO) return: {m:.1f} ± {s:.1f}")
    (MODEL_DIR / f"simopt_phi_seed{seed}.json").write_text(json.dumps(phi, indent=2))
    return final_path

# -------------------------------------------------------------------------
#  main
# -------------------------------------------------------------------------

def main(seed: int = 42, device: str = "cpu"):
    set_seed(seed)
    train_vanilla(SOURCE_ENV, "source_base", seed, device)
    train_vanilla(TARGET_ENV, "target_base", seed, device)
    train_simopt(seed, device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    main(args.seed, args.device)



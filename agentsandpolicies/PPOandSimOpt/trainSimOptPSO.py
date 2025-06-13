"""
trainSimOptPSO.py
-----------------
Addestra tre modelli:

1. source_base_seed<seed>_final.zip   – PPO vanilla su source
2. target_base_seed<seed>_final.zip   – PPO vanilla su target
3. simopt_final_seed<seed>.zip        – PPO addestrato con SimOpt (ottimizzazione masse via PSO)

Esempio d'uso:
$ python trainSimOptPSO.py --seed 42 --device cpu
"""

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List

import gym
import nevergrad as ng
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor

# ---------------------------------------------------------------------------
#  wrapped utilities (no external utils_simopt.py needed)
# ---------------------------------------------------------------------------
class HopperMassRandomGaussianWrapper(gym.Wrapper):
    """Campiona le prime 3 masse (torso/thigh, leg, foot) da N(μ, σ²) ad
    ogni reset, usando il dizionario *phi* {part: [μ, σ]} passato a init.
    """

    def __init__(self, env: gym.Env, phi: Dict[str, List[float]]):
        super().__init__(env)
        self.phi = phi  # dict con chiavi 'thigh', 'leg', 'foot'
        self._order = ["thigh", "leg", "foot"]  # prime 3 masse di body_mass[1:]

    def reset(self, **kwargs):
        # preleva masse originali e sostituisce le prime 3 con campioni
        masses = self.env.get_parameters().copy()
        for i, key in enumerate(self._order):
            mu, sigma = self.phi[key]
            masses[i] = np.random.normal(mu, sigma)
        self.env.set_parameters(masses)
        return self.env.reset(**kwargs)

    def step(self, action):
        return self.env.step(action)


def gap(real: List[np.ndarray], sim: List[np.ndarray]) -> float:
    """Gap semplice: somma degli L2 tra vettori osservazione.
    *real* e *sim* sono liste di array già tagliati alla stessa lunghezza.
    """
    real_arr = np.concatenate(real)
    sim_arr = np.concatenate(sim)
    return float(np.linalg.norm(real_arr - sim_arr, ord=2))


def get_obs(model: PPO, env: gym.Env, n_episodes: int = 3) -> List[np.ndarray]:
    """Raccoglie le osservazioni di *n_episodes* esecuzioni deterministiche."""
    obs_trajs: List[np.ndarray] = []
    for _ in range(n_episodes):
        done, traj = False, []
        obs = env.reset()
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, done, _ = env.step(action)
            traj.append(obs)
        obs_trajs.append(np.array(traj).flatten())
    env.close()
    return obs_trajs

# ---------------------------------------------------------------------------
#  costanti
# ---------------------------------------------------------------------------
SOURCE_ENV = "CustomHopper-source-v0"
TARGET_ENV = "CustomHopper-target-v0"
TOTAL_STEPS_BASE = 2_000
SIMOPT_PSO_BUDGET = 40
SIMOPT_PSO_SIGMA = 0.5
FINE_TUNE_STEPS = 40_000
FINAL_TRAIN_STEPS = 2_000_000
BETA = 0.1
MODEL_DIR = Path("models_weights"); MODEL_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
#  util varie
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
#  SimOpt con PSO
# ---------------------------------------------------------------------------

def train_simopt(seed: int, device: str):
    # distribuzione iniziale delle masse (μ, σ)
    phi = {"thigh": [3.93, 0.5], "leg": [2.71, 0.5], "foot": [5.09, 0.5]}
    model = PPO("MlpPolicy", make_env(SOURCE_ENV, seed, phi), seed=seed, verbose=0, device=device)

    instrum = ng.p.Dict(
        thigh=ng.p.Scalar(phi["thigh"][0]).set_bounds(0.1, 20).set_mutation(sigma=SIMOPT_PSO_SIGMA),
        leg  =ng.p.Scalar(phi["leg"][0]).set_bounds(0.1, 20).set_mutation(sigma=SIMOPT_PSO_SIGMA),
        foot =ng.p.Scalar(phi["foot"][0]).set_bounds(0.1, 20).set_mutation(sigma=SIMOPT_PSO_SIGMA),
    )
    optim = ng.optimizers.PSO(parametrization=instrum, budget=SIMOPT_PSO_BUDGET)

    def evaluate_phi(phi_cand: Dict[str, List[float]]) -> float:
        # 1) fine‑tune breve
        model.set_env(make_env(SOURCE_ENV, seed, phi_cand))
        model.learn(total_timesteps=FINE_TUNE_STEPS, reset_num_timesteps=False)
        # 2) gap osservazioni
        real = get_obs(model, make_env(TARGET_ENV, seed), 3)
        sim  = get_obs(model, make_env(SOURCE_ENV, seed, phi_cand), 3)
        L = min(min(len(r) for r in real), min(len(s) for s in sim))
        gap_val = gap([r[:L] for r in real], [s[:L] for s in sim])
        # 3) reward target
        ret, _ = evaluate_policy(model, make_env(TARGET_ENV, seed), 3, True)
        return gap_val - BETA * ret

    # PSO loop
    for k in range(SIMOPT_PSO_BUDGET):
        cand = optim.ask()
        mu_dict = {key: [cand[key].value, phi[key][1]] for key in phi}
        loss = evaluate_phi(mu_dict)
        optim.tell(cand, loss)
        print(f"PSO iter {k}: loss {loss:.3f}")

    best_mu = optim.recommend().value
    print("PSO – migliori μ:", best_mu)
    for key in phi:
        phi[key][0] = best_mu[key]

    # training finale lungo
    final_path = MODEL_DIR / f"simopt_final_seed{seed}.zip"
    train_env = make_env(SOURCE_ENV, seed, phi)
    final_model = PPO("MlpPolicy", train_env, seed=seed, verbose=0, device=device)
    final_model.learn(total_timesteps=FINAL_TRAIN_STEPS)
    final_model.save(str(final_path)); train_env.close()
    m, s = evaluate_policy(final_model, make_env(SOURCE_ENV, seed, phi), 20, True)
    print(f"SimOpt (PSO) return: {m:.1f} ± {s:.1f}")

    # salva φ
    (MODEL_DIR / f"simopt_phi_seed{seed}.json").write_text(json.dumps(phi, indent=2))
    return final_path

# ---------------------------------------------------------------------------
#  main pipeline
# ---------------------------------------------------------------------------

def main(seed: int = 42, device: str = "cpu"):
    set_seed(seed)
    train_vanilla(SOURCE_ENV, "source_base", seed, device)
    train_vanilla(TARGET_ENV, "target_base", seed, device)
    train_simopt(seed, device)


# ---------------- CLI ----------------
if __name__ == "__main__":
    cl = argparse.ArgumentParser()
    cl.add_argument("--seed", type=int, default=42)
    cl.add_argument("--device", default="cpu")
    args = cl.parse_args()
    main(args.seed, args.device)


import gym
import numpy as np
import random
import torch
from itertools import product
from stable_baselines3 import PPO
from stable_baselines3.common.utils import get_linear_fn
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from env.custom_hopper import *

# 1) Define a single global seed
SEED = 42

# 2) Fix seeds for all libraries
np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# 3) Linear learning-rate schedule
lr_schedule = get_linear_fn(start=1e-4, end=0.0, end_fraction=1.0)

# 4) Hyperparameter grid
param_grid = {
    "n_steps":       [4096], #2048, 4096, 8192
    "batch_size":    [32, 128], #32, 128
    "gae_lambda":    [0.8, 0.9], #0.8, 0.9
    "gamma":         [0.99],  #0.95, 0.99
    "n_epochs":      [10,20], #10,20
    "clip_range":    [0.2], 
    "ent_coef":      [0.0, 0.005], #0.0, 0.005
    "vf_coef":       [0.5, 1.0], #0.5, 1.0
    "max_grad_norm": [0.5, 1.0] #0.5, 1.0
}


# 5) Utility to create a seeded, normalized VecEnv
def make_env(env_id: str, seed: int):
    env = gym.make(env_id)
    env.seed(seed)
    vec = DummyVecEnv([lambda: env])
    return VecNormalize(vec, norm_obs=True, norm_reward=True)

if __name__ == "__main__":
    # 6) Prepare combinations
    combinations = list(product(*param_grid.values()))

    best_score = -float('inf')
    best_params = None

    for vals in combinations:
        # 6.1) Map each hyperparameter to its test value
        hp = dict(zip(param_grid.keys(), vals))
        print(f"[Seed {SEED}] Testing: {hp}")

        # 6.2) Re-fissiamo i seed prima di ogni trial
        np.random.seed(SEED)
        random.seed(SEED)
        torch.manual_seed(SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(SEED)

        # 6.3) Crea training ed eval env
        train_vec = make_env("CustomHopper-source-v0", seed=SEED)
        eval_vec  = make_env("CustomHopper-source-v0", seed=SEED + 1)

        # 6.4) Instanzia PPO
        model = PPO(
            policy='MlpPolicy',           # MLP policy network
            env=train_vec,                # vectorized + normalized train env
            seed=SEED,                    # random seed
            verbose=0,                    # silent
            n_steps=hp["n_steps"],        # rollout steps
            batch_size=hp["batch_size"],  # minibatch size
            gae_lambda=hp["gae_lambda"],  # GAE-λ
            gamma=hp["gamma"],            # discount factor
            n_epochs=hp["n_epochs"],      # optimization epochs
            clip_range=hp["clip_range"],  # PPO clip parameter
            ent_coef=hp["ent_coef"],      # entropy bonus coefficient
            vf_coef=hp["vf_coef"],        # value-function loss coeff
            max_grad_norm=hp["max_grad_norm"],  # gradient clipping
            learning_rate=lr_schedule     # linear lr schedule
        )

        # 6.5) Train
        model.learn(total_timesteps=200_000)

        # 6.6) Evaluate deterministically
        mean_reward, _ = evaluate_policy(
            model, eval_vec,
            n_eval_episodes=5,
            deterministic=True
        )
        print(f"→ Mean reward: {mean_reward:.1f}")

        # 6.7) Track the best
        if mean_reward > best_score:
            best_score, best_params = mean_reward, hp

    # 7) Print final result
    print("\n=== Best hyperparameters ===")
    for k, v in best_params.items():
        print(f"{k:15s}: {v}")
    print(f"Best mean reward : {best_score:.2f}")

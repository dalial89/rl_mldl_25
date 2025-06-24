import gym
import itertools
import numpy as np
import random
import torch

from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.utils import get_linear_fn
from stable_baselines3.common.monitor import Monitor

from env.custom_hopper import *

# 1) Utility to create a seeded, normalized VecEnv
def make_env(env_id: str, seed: int):
    env = Monitor(gym.make(env_id), filename=None)
    env.seed(seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)
    env.reset()
    return env




def main():
    # 2) Define a single global seed
    SEED = 42
    np.random.seed(SEED)
    random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    print("")
    print("----- The tuning of the hyperparameters for the PPO just started! -----")
    print("")

    # 3) Linear learning-rate schedule
    lr_schedule = get_linear_fn(start=1e-4, end=0.0, end_fraction=1.0)

    # 4) Hyperparameter grid
    param_grid = {
        "gamma":         [0.95, 0.98,0.99],
        "learning_rate": [lr_schedule, 1e-3, 1e-4, 3e-4, 6e-4, 9e-4, 1e-6, 1e-8]

    }

    # 5) Prepare combinations
    combos     = list(itertools.product(*param_grid.values()))
    best_score = -float("inf")
    best_params= None

    for vals in combos:
        hp = dict(zip(param_grid.keys(), vals))
        print(f"[Seed {SEED}] Testing: {hp}")

        # 6) Reset seeds before each run
        np.random.seed(SEED)
        random.seed(SEED)
        torch.manual_seed(SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(SEED)

        # 7) Create train & eval envs
        train_vec = make_env("CustomHopper-source-v0", seed=SEED)
        eval_vec  = make_env("CustomHopper-source-v0", seed=SEED + 1)

        # 8) Instantiate PPO
        model = PPO(
            policy='MlpPolicy',
            env=train_vec,
            seed=SEED,
            verbose=0,
            gamma=hp["gamma"],
            learning_rate=hp["learning_rate"]
        )

        # 9) Train
        model.learn(total_timesteps=1_000_000)

        # 10) Evaluate
        mean_reward, _ = evaluate_policy(
            model, eval_vec,
            n_eval_episodes=40,
            deterministic=True
        )
        print(f"→ Mean reward: {mean_reward:.1f}")

        # 11) Track best
        if mean_reward > best_score:
            best_score, best_params = mean_reward, hp

    # 13) Print final result
    print("\n=== Best hyperparameters ===")
    for k, v in best_params.items():
        print(f"{k:15s}: {v}")
    print(f"Best mean reward : {best_score:.2f}")

if __name__ == "__main__":
    main()


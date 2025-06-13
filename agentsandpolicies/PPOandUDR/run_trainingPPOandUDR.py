import gym
import argparse
import csv
import os
from pathlib import Path
import numpy as np
import random
import torch

from env.custom_hopper import *
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import get_linear_fn
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

# 1) Global seed
def set_seeds(seed: int):
    global SEED
    SEED = seed
    np.random.seed(seed); random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# 2) Mass-randomization wrapper (UDR)
class HopperMassRandomWrapper(gym.Wrapper):
    """
    Multiply masses of thigh(2), leg(3), foot(4) by a random factor
    drawn in the provided ranges at every reset.
    """
    def __init__(self, env, ranges):
        super().__init__(env)
        self.base_mass = env.sim.model.body_mass.copy()
        self.ranges    = ranges

    def reset(self, **kwargs):
        self.env.sim.model.body_mass[:] = self.base_mass
        for idx, (low, high) in self.ranges.items():
            self.env.sim.model.body_mass[idx] *= np.random.uniform(low, high)
        return self.env.reset(**kwargs)

# 3) Best UDR bounds found in the sweep (replace if different)
BEST_UDR_RANGES = {2: (0.70, 1.30), 3: (0.90, 1.10), 4: (0.50, 1.50)}

# 4) Project-wide folders
SCRIPT_DIR  = Path(__file__).resolve().parent
PROJ_ROOT   = SCRIPT_DIR.parents[1]                   # rl_mldl_25/
DATA_DIR    = PROJ_ROOT / "training_data"
WEIGHTS_DIR = PROJ_ROOT / "models_weights"
DATA_DIR.mkdir(exist_ok=True)
WEIGHTS_DIR.mkdir(exist_ok=True)

# 5) Save per-episode reward + running variance to CSV
def save_rewards_csv(monitor_env, csv_path):
    rewards = monitor_env.get_episode_rewards()
    if not rewards:
        print("No rewards recorded – skipping CSV.")
        return
    cumulative = []
    running_var = []
    for r in rewards:
        cumulative.append(r)
        running_var.append(np.var(cumulative))
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["episode", "reward", "running_variance"])
        for ep, (r, v) in enumerate(zip(rewards, running_var), 1):
            writer.writerow([ep, r, v])
    print(f"Saved reward log to {csv_path}")

# 6) Build a VecNormalize environment (UDR optional on source)
def make_vec_env(env_id, seed, use_udr):
    def _init():
        env = gym.make(env_id)
        if env_id == "CustomHopper-source-v0" and use_udr:
            env = HopperMassRandomWrapper(env, BEST_UDR_RANGES)
        env.seed(seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        env = Monitor(env, filename=None)
        return env
    vec = DummyVecEnv([_init])
    return VecNormalize(vec, norm_obs=True, norm_reward=True)

# 7) Training routine
def train_and_save(env_id, run_tag, seed, use_udr=False):
    log_dir   = SCRIPT_DIR / f"logs_{run_tag}_seed{seed}"
    csv_file  = DATA_DIR  / f"{run_tag}_seed{seed}_rewards.csv"
    model_file= WEIGHTS_DIR / f"{run_tag}_seed{seed}_final.zip"

    print(f"\n=== Training {env_id} (UDR={use_udr}) ===")

    train_vec = make_vec_env(env_id, seed,     use_udr)
    eval_vec  = make_vec_env(env_id, seed + 1, False)
    eval_vec.norm_reward = False

    check_env(train_vec.envs[0])

    lr_sched = get_linear_fn(start=1e-4, end=0.0, end_fraction=1.0)
    model = PPO(
        "MlpPolicy",
        train_vec,
        seed=SEED,
        verbose=0,
        n_steps=8192,
        batch_size=32,
        gae_lambda=0.9,
        gamma=0.95,
        n_epochs=10,
        clip_range=0.2,
        ent_coef=0.0,
        vf_coef=1.0,
        max_grad_norm=1.0,
        learning_rate=lr_sched,
    )

    eval_cb = EvalCallback(
        eval_vec,
        best_model_save_path=log_dir / "best_model",
        log_path=log_dir,
        eval_freq=50_000,
        n_eval_episodes=10,
        deterministic=True,
    )

    model.learn(total_timesteps=2_000_000, callback=eval_cb)

    model.save(model_file)
    train_vec.save(log_dir / "vecnormalize.pkl")
    print(f"Model saved to {model_file}")

    mean_ret, std_ret = evaluate_policy(
        model, eval_vec, n_eval_episodes=100, deterministic=True  # same as tuning
    )
    print(f"Final deterministic return: {mean_ret:.2f} ± {std_ret:.2f}")

    save_rewards_csv(train_vec.envs[0], csv_file)

# 8) entry-point 
def main(seed: int, use_udr: bool) -> None:
    set_seeds(seed)                                    


    tag_source = "source_udr" if use_udr else "source_base"
    train_and_save("CustomHopper-source-v0",
                   run_tag=tag_source,
                   seed=seed,
                   use_udr=use_udr)


    train_and_save("CustomHopper-target-v0",
                   run_tag="target_base",
                   seed=seed,
                   use_udr=False)

if __name__ == "__main__":
    main()

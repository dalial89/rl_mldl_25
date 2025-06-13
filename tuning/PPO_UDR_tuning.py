import gym
import itertools
import numpy as np
import random
import torch

from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import get_linear_fn
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from env.custom_hopper import *

# 1) Define the mass-randomization wrapper
class HopperMassRandomWrapper(gym.Wrapper):
    """
    On first reset saves the factory masses.
    On each subsequent reset multiplies specified bodies by a random factor
    drawn uniformly in the provided range.
    Body indices for Hopper MuJoCo: 2=thigh, 3=leg, 4=foot.
    """
    def __init__(self, env, ranges):
        super().__init__(env)
        self.base_mass = env.sim.model.body_mass.copy()
        self.ranges = ranges

    def reset(self, **kwargs):
        # 1.1) Restore original masses
        self.env.sim.model.body_mass[:] = self.base_mass
        # 1.2) Apply random factors to specified links
        for idx, (low, high) in self.ranges.items():
            factor = np.random.uniform(low, high)
            self.env.sim.model.body_mass[idx] *= factor
        return self.env.reset(**kwargs)

    def get_parameters(self):
        return self.env.sim.model.body_mass.copy()

def main():
    # 2) Set a single global seed for reproducibility
    SEED = 42
    np.random.seed(SEED)
    random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    print("\n----- The tuning of the bounds of the masses for the UDR using PPO just started! -----\n")

    # 3) Define the learning-rate schedule and PPO hyperparameters
    lr_schedule = get_linear_fn(start=1e-4, end=0.0, end_fraction=1.0)
    PPO_KWARGS = dict(
        policy="MlpPolicy",
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
        learning_rate=lr_schedule,
    )

    # 4) Define the percentage ranges and build the sweep list
    PERCENTS     = [10, 30, 50]
    THIGH_RANGES = [(1 - p/100, 1 + p/100) for p in PERCENTS]
    LEG_RANGES   = [(1 - p/100, 1 + p/100) for p in PERCENTS]
    FOOT_RANGES  = [(1 - p/100, 1 + p/100) for p in PERCENTS]

    MASS_SWEEP = [
        {2: t, 3: l, 4: f}
        for t, l, f in itertools.product(THIGH_RANGES, LEG_RANGES, FOOT_RANGES)
    ]

    # 5) Helper to create a seeded, normalized VecEnv with mass-randomization
    def make_env(env_id: str, seed: int, mass_ranges):
        def _init():
            env = Monitor(gym.make(env_id), filename=None)
            env = HopperMassRandomWrapper(env, mass_ranges)
            env.seed(seed)
            env.action_space.seed(seed)
            env.observation_space.seed(seed)
            env.reset()
            return env
        vec = DummyVecEnv([_init])
        return VecNormalize(vec, norm_obs=True, norm_reward=True)

    # 6) Main sweep: train and evaluate for each mass configuration
    best_score  = -float("inf")
    best_ranges = None

    for ranges in MASS_SWEEP:
        print(f"[Seed {SEED}] Testing mass ranges: {ranges}")

        # 6.1) Reset seeds before each run
        np.random.seed(SEED)
        random.seed(SEED)
        torch.manual_seed(SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(SEED)

        # 6.2) Create training and evaluation environments
        train_vec = make_env("CustomHopper-source-v0", SEED,     ranges)
        eval_vec  = make_env("CustomHopper-source-v0", SEED + 1, ranges)

        # 6.3) Instantiate and train the PPO model
        model = PPO(env=train_vec, **PPO_KWARGS)
        model.learn(total_timesteps=200_000)

        # 6.4) Freeze VecNormalize statistics and prepare eval
        train_vec.training     = False
        eval_vec.training      = False
        eval_vec.obs_rms       = train_vec.obs_rms
        eval_vec.ret_rms       = train_vec.ret_rms
        eval_vec.norm_reward   = False   # use real rewards for eval

        # 6.5) Evaluate the policy
        mean_reward, _ = evaluate_policy(
            model,
            eval_vec,
            n_eval_episodes=40,
            deterministic=True
        )
        print(f"→ Mean reward: {mean_reward:.1f}\n")

        # 6.6) Track the best-performing mass ranges
        if mean_reward > best_score:
            best_score, best_ranges = mean_reward, ranges

    # 7) Print the overall best mass ranges and score
    print("=== Best randomized mass ranges ===")
    for idx in sorted(best_ranges):
        low, high = best_ranges[idx]
        print(f"body {idx}: [{low:.2f}, {high:.2f}]")
    print(f"Best mean reward: {best_score:.2f}")

if __name__ == "__main__":
    main()


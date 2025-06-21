import gym
import itertools
import numpy as np
import random
import torch

from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.utils import get_linear_fn
from stable_baselines3.common.monitor import Monitor

from env.custom_hopper import *  # registers CustomHopper-source-v0/target-v0

# 1) Utility to create a seeded Monitor-wrapped env with UDR
class HopperMassRandomWrapper(gym.Wrapper):
    """
    On each reset multiplies specified bodies by a random factor
    drawn uniformly in provided ranges.
    Body indices: 2=thigh, 3=leg, 4=foot.
    """
    def __init__(self, env, mass_ranges):
        super().__init__(env)
        self.base_mass = env.sim.model.body_mass.copy()
        self.ranges    = mass_ranges

    def reset(self, **kwargs):
        # restore nominal masses
        self.env.sim.model.body_mass[:] = self.base_mass
        # apply random factors
        for idx, (low, high) in self.ranges.items():
            self.env.sim.model.body_mass[idx] *= np.random.uniform(low, high)
        return self.env.reset(**kwargs)


def make_env(env_id: str, seed: int, mass_ranges: dict):
    env = Monitor(gym.make(env_id), filename=None)
    env = HopperMassRandomWrapper(env, mass_ranges)
    env.seed(seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)
    return env


def main():
    # 2) Define a single global seed
    SEED = 42
    np.random.seed(SEED)
    random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    print("\n----- UDR hyperparameter sweep for PPO -----\n")

    # 4) Define PPO fixed args and UDR sweep grid
    # fixed PPO settings from prior tuning
    PPO_KWARGS = dict(
        policy       = 'MlpPolicy',
        seed         = SEED,
        verbose      = 0,
        gamma        = 0.99,
        learning_rate= 3e-4
    )
    # UDR parameter grid: percentage perturbations
    PERCENTS      = [10, 30, 50]
    THIGH_RANGE   = [(1-p/100,1+p/100) for p in PERCENTS]
    LEG_RANGE     = [(1-p/100,1+p/100) for p in PERCENTS]
    FOOT_RANGE    = [(1-p/100,1+p/100) for p in PERCENTS]

    param_grid = {
        2: THIGH_RANGE,
        3: LEG_RANGE,
        4: FOOT_RANGE,
    }

    # 5) Prepare combinations of mass_ranges
    combos     = []
    for t, l, f in itertools.product(param_grid[2], param_grid[3], param_grid[4]):
        combos.append({2: t, 3: l, 4: f})

    best_score  = -float('inf')
    best_ranges = None

    # 6) Loop over mass-range combos
    for mass_ranges in combos:
        print(f"[Seed {SEED}] Testing mass ranges: {mass_ranges}")
        # reset seeds per run
        np.random.seed(SEED); random.seed(SEED); torch.manual_seed(SEED)
        if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)

        # create envs
        env_id   = 'CustomHopper-source-v0'
        train_env= make_env(env_id, SEED,     mass_ranges)
        eval_env = make_env(env_id, SEED + 1, mass_ranges)

        # 7) Instantiate PPO with fixed best params
        model = PPO(env=train_env, **PPO_KWARGS)
        model.learn(total_timesteps=1_000_000)

        # 8) Evaluate
        mean_reward, _ = evaluate_policy(
            model,
            eval_env,
            n_eval_episodes=40,
            deterministic=True
        )
        print(f"→ Mean reward: {mean_reward:.1f}\n")

        # 9) Track best
        if mean_reward > best_score:
            best_score, best_ranges = mean_reward, mass_ranges

    # 10) Print final result
    print("=== Best randomized mass ranges ===")
    for idx, (lo, hi) in best_ranges.items():
        print(f"body {idx}: [{lo:.2f}, {hi:.2f}]")
    print(f"Best mean reward: {best_score:.2f}")


if __name__ == '__main__':
    main()


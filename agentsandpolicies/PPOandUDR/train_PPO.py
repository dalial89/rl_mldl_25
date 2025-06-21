import os, csv, argparse, random, numpy as np, torch, gym
from pathlib import Path
from stable_baselines3 import PPO
from env.custom_hopper import *

from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.utils import get_linear_fn

class HopperMassRandomWrapper(gym.Wrapper):
    """
    At every reset multiply the masses of thigh(2), leg(3), foot(4)
    by a random factor drawn in the specified ranges.
    """
    def __init__(self, env, ranges):
        super().__init__(env)
        self.base_mass = env.sim.model.body_mass.copy()
        self.ranges    = ranges            # dict {body_id: (low, high)}

    def reset(self, **kwargs):
        # restore nominal masses
        self.env.sim.model.body_mass[:] = self.base_mass
        # apply random factors
        for idx, (lo, hi) in self.ranges.items():
            self.env.sim.model.body_mass[idx] *= np.random.uniform(lo, hi)
        return self.env.reset(**kwargs)


def make_env(env_id: str, seed: int, use_udr: bool) -> Monitor:
    """Create the environment, wrap with Monitor and (optionally) with UDR."""
    env = gym.make(env_id)

    if use_udr:
        """
        BEST RANGES:

        body 2: [0.90, 1.10]
        body 3: [0.90, 1.10]
        body 4: [0.50, 1.50]
        """
        ranges = {2: (0.9, 1.1), 3: (0.9, 1.1), 4: (0.7, 1.3)}
        env = HopperMassRandomWrapper(env, ranges)

    env.seed(seed)
    env.action_space.seed(seed)
    return Monitor(env)

def save_rewards_csv(monitor_env: Monitor, csv_path: str) -> None:
    """Dump per-episode returns and running variance to a CSV file."""
    rewards = monitor_env.get_episode_rewards()
    if len(rewards) == 0:
        print("[WARN] No episodes recorded – CSV not written.")
        return

    cumulative, running_var = [], []
    for r in rewards:
        cumulative.append(r)
        running_var.append(np.var(cumulative))

    with open(csv_path, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["episode", "return", "running_variance"])
        for ep, (r, v) in enumerate(zip(rewards, running_var), 1):
            wr.writerow([ep, r, v])
    print(f"Saved reward log → {csv_path}")


def train(env_id: str, seed: int, total_ts: int, device: str, use_udr: bool):
    env_tag = "source" if "source" in env_id else "target"

    env  = make_env(env_id, seed, use_udr)
    lr_fn = get_linear_fn(start=1e-4, end=0.0, end_fraction=1.0)   # linear LR decay

    """
    BEST PARAMETERS
    gamma          : 0.99
    learning_rate  : 3e-4
    """



    model = PPO(
        policy="MlpPolicy",
        env=env,
        seed=seed,
        device=device,
        verbose=1,
        # best parameters
        gamma=0.99,
        learning_rate=3e-4   
    )



    model.learn(total_timesteps=total_ts)

    # save
    weights_dir = Path(__file__).resolve().parents[2] / "models_weights"
    data_dir    = Path(__file__).resolve().parents[2] / "models_data"
    weights_dir.mkdir(exist_ok=True)
    data_dir.mkdir(exist_ok=True)

    weight_file = weights_dir / f"ppo_tuned_{env_tag}_seed_{seed}_UDR_{use_udr}.zip"
    model.save(str(weight_file))
    print(f"Model weights saved → {weight_file}")

    csv_file = data_dir / f"ppo_tuned_{env_tag}_seed_{seed}_UDR_{use_udr}_returns.csv"
    save_rewards_csv(env, str(csv_file))

    # quick evaluation
    mean_r, std_r = evaluate_policy(model, env, n_eval_episodes=10, deterministic=True)
    print(f"Average return on {env_tag}: {mean_r:.1f} ± {std_r:.1f}")

    env.close()

def parse_args():
    p = argparse.ArgumentParser(description="PPO trainer for CustomHopper")
    p.add_argument("--seed",      type=int, default=42, help="Random seed")
    p.add_argument("--timesteps", type=int, default=2_000_000,
                   help="Number of training timesteps")
    p.add_argument("--env",       choices=["source", "target"], default="source",
                   help="Which environment to train on")
    p.add_argument("--udr",       action="store_true",
                   help="Enable mass-randomization (UDR)")
    p.add_argument("--device",    default="cpu",
                   help="Computation device (cpu | cuda | cuda:0 …)")
    return p.parse_args()


def main():
    args = parse_args()

    # global seeding
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    env_id = "CustomHopper-source-v0" if args.env == "source" else "CustomHopper-target-v0"
    train(env_id, args.seed, args.timesteps, args.device, args.udr)

if __name__ == "__main__":
    main()

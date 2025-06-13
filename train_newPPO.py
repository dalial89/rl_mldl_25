import argparse
import random
import numpy as np
import torch
import gym
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from env.custom_hopper import * 

# --------------------------------------------------------------------
# Costanti
# --------------------------------------------------------------------
SEED = 42
TOTAL_TIMESTEPS = 2_000_000          # come nello script “grande”
SOURCE_ENV_ID  = "CustomHopper-source-v0"
TARGET_ENV_ID  = "CustomHopper-target-v0"
SOURCE_MODEL   = "modelPPO_source"
TARGET_MODEL   = "modelPPO_target"

# --------------------------------------------------------------------
# Utility
# --------------------------------------------------------------------
def make_env(env_id: str, seed: int):
    """Crea l’ambiente e applica il seed."""
    env = gym.make(env_id)
    env.seed(seed)
    env.action_space.seed(seed)
    return env

def train_one(env_id: str, model_name: str, device: str):
    """Allena PPO su un singolo ambiente e salva il modello."""
    print(f"\n=== Training su {env_id} ===")
    env = make_env(env_id, SEED)

    model = PPO(
        policy="MlpPolicy",
        env=env,
        verbose=1,
        device=device,
        seed=SEED,
    )

    model.learn(total_timesteps=TOTAL_TIMESTEPS)
    model.save(model_name)

    # Valutazione veloce (10 episodi)
    mean_r, std_r = evaluate_policy(model, env, n_eval_episodes=10, deterministic=True)
    print(f"Return medio su {env_id}: {mean_r:.1f} ± {std_r:.1f}")

    env.close()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--device", default="cpu",
        help="Dispositivo di calcolo (cpu, cuda, cuda:0, …)"
    )
    return parser.parse_args()

def main():
    # fissiamo i seed a livello globale
    np.random.seed(SEED)
    random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    args = parse_args()

    # allenamento su source e target
    train_one(SOURCE_ENV_ID, SOURCE_MODEL, args.device)
    train_one(TARGET_ENV_ID, TARGET_MODEL, args.device)

if __name__ == "__main__":
    main()

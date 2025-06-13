"""
Valuta i modelli PPO (modelPPO_source.zip e modelPPO_target.zip) sulle
varie combinazioni di ambienti CustomHopper-source/target.

• Nessun VecNormalize/UDR
• Seed fisso per riproducibilità
"""

import argparse
import random
import numpy as np
import torch
import gym
import os

from env.custom_hopper import *            # noqa: F401  (registra gli env)

from stable_baselines3 import PPO

# --------------------------------------------------------------------
# Costanti e seed
# --------------------------------------------------------------------
SEED = 42
np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

MODEL_SOURCE = "modelPPO_source.zip"   # salvati dallo script di training
MODEL_TARGET = "modelPPO_target.zip"

# --------------------------------------------------------------------
# Argomenti CLI
# --------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Evaluate PPO models on CustomHopper")
    p.add_argument(
        "--device", default="cpu",
        help="cpu, cuda o cuda:0 …"
    )
    p.add_argument(
        "--episodes", type=int, default=50,
        help="Episodi di test per ogni caso"
    )
    p.add_argument(
        "--render", action="store_true",
        help="Mostra l'ambiente mentre si testa"
    )
    # Possibilità di saltare o aggiungere casi via riga di comando
    p.add_argument(
        "--add-target-source", action="store_true",
        help="Valuta anche target→source"
    )
    return p.parse_args()

# --------------------------------------------------------------------
# Funzione di test
# --------------------------------------------------------------------
def test_case(model_path: str, env_id: str, episodes: int, render: bool, device: str):
    """Restituisce (mean_return, std_return)."""
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Modello non trovato: {model_path}")

    env = gym.make(env_id)
    env.seed(SEED)
    env.action_space.seed(SEED)

    model: PPO = PPO.load(model_path, device=device)

    returns = []
    for ep in range(episodes):
        obs, done, ep_ret = env.reset(), False, 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _ = env.step(action)
            ep_ret += float(reward)
            if render:
                env.render()
        returns.append(ep_ret)

    env.close()
    return np.mean(returns), np.std(returns)

# --------------------------------------------------------------------
# Main
# --------------------------------------------------------------------
def main() -> None:
    args = parse_args()

    test_cases = [
        ("source→source", MODEL_SOURCE, "CustomHopper-source-v0"),
        ("source→target", MODEL_SOURCE, "CustomHopper-target-v0"),
        ("target→target", MODEL_TARGET, "CustomHopper-target-v0"),
    ]
    if args.add_target_source:
        test_cases.append(
            ("target→source", MODEL_TARGET, "CustomHopper-source-v0")
        )

    print(f"\nEpisodi per caso: {args.episodes}  |  Render: {args.render}\n")

    for label, model_path, env_id in test_cases:
        mean_ret, std_ret = test_case(
            model_path, env_id,
            episodes=args.episodes,
            render=args.render,
            device=args.device
        )
        print(f"{label:<15}  Env: {env_id:<25}  "
              f"Return medio: {mean_ret:8.2f} ± {std_ret:.2f}")

# --------------------------------------------------------------------
if __name__ == "__main__":
    main()


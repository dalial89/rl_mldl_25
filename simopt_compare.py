"""
SimOpt con confronto di ottimizzatori Nevergrad
----------------------------------------------
• Policy learner: PPO (Stable-Baselines3)
• Environment: CustomHopper-source/target
• Obiettivo: minimizzare la discrepanza (reward_target – reward_source in valore assoluto)
• Algoritmi testati: CMA, SepCMA, DE, PSO, NGOpt8
"""

import argparse, random, time, json, warnings
from pathlib import Path

import gym
import numpy as np
import pandas as pd
import torch
import nevergrad as ng
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy

from env.custom_hopper import CustomHopper   # Assicurati che il tuo package sia importabile

# -------------------------- parametri globali --------------------------
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
warnings.filterwarnings("ignore", category=UserWarning)      # silenzia Monitor warning SB3

# -------------------------- helper funzione RL -------------------------
def train_policy(env: gym.Env, total_timesteps: int = 5_000) -> PPO:
    """Addestra PPO sull'env fornito e restituisce il modello."""
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=1e-3,
        gamma=0.99,
        seed=SEED,
        verbose=0,
    )
    model.learn(total_timesteps=total_timesteps)
    return model


def eval_reward(model: PPO, env_id: str, masses: np.ndarray, episodes: int) -> float:
    """Ritorna il reward medio su *episodes* episodi nell'ambiente dato."""
    env = gym.make(env_id)
    env.set_parameters(masses)
    mean, _ = evaluate_policy(model, env, n_eval_episodes=episodes, render=False)
    env.close()
    return mean


# -------------------------- SimOpt loop singolo ------------------------
def simopt_once(
    optim_cls: type,
    budget: int,
    episodes_eval: int,
    init_mu=(3.92699082, 2.71433605, 5.0893801),
    init_sigma: float = 0.5,
) -> dict:
    """Esegue una singola ottimizzazione SimOpt con l'algoritmo scelto."""
    # Nevergrad parametrization (3 masse)
    instrument = ng.p.Dict(
        x1=ng.p.Scalar(init_mu[0]).set_mutation(sigma=init_sigma),
        x2=ng.p.Scalar(init_mu[1]).set_mutation(sigma=init_sigma),
        x3=ng.p.Scalar(init_mu[2]).set_mutation(sigma=init_sigma),
    )
    optim = optim_cls(parametrization=instrument, budget=budget)

    best_disc = np.inf
    best_masses = None

    for _ in range(budget):
        x = optim.ask()
        masses = np.array([x["x1"], x["x2"], x["x3"]])

        # -- train PPO sulla source con queste masse
        env_src = gym.make("CustomHopper-source-v0")
        env_src.set_parameters(masses)
        model = train_policy(env_src, total_timesteps=5_000)
        env_src.close()

        # -- calcola discrepanza: |reward_target - reward_source|
        r_target = eval_reward(model, "CustomHopper-target-v0", masses, episodes_eval)
        r_source = eval_reward(model, "CustomHopper-source-v0",  masses, episodes_eval)
        discrepancy = abs(r_target - r_source)

        optim.tell(x, discrepancy)

        if discrepancy < best_disc:
            best_disc = discrepancy
            best_masses = masses.copy()

    recommendation = optim.recommend()
    return {
        "best_disc": best_disc,
        "best_masses": best_masses.tolist(),
        "ng_value": recommendation.value,   # dict x1,x2,x3
    }


# ----------------------------- main ------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=3, help="ripetizioni indipendenti per algoritmo")
    parser.add_argument("--budget", type=int, default=400, help="valutazioni per ottimizzatore")
    parser.add_argument("--episodes", type=int, default=30, help="episodi di test reward")
    args = parser.parse_args()

    ALGOS = {
        "CMA": ng.optimizers.CMA,
        "SepCMA": ng.optimizers.SepCMA,
        "DE": ng.optimizers.DE,
        "PSO": ng.optimizers.PSO,
        "NGOpt8": ng.optimizers.NGOpt8,
    }

    rows = []
    t0 = time.time()

    for name, cls in ALGOS.items():
        for run in range(args.runs):
            print(f"[{name}] run {run+1}/{args.runs}")
            out = simopt_once(
                optim_cls=cls,
                budget=args.budget,
                episodes_eval=args.episodes,
            )
            out.update(algo=name, run=run)
            rows.append(out)

    df = pd.DataFrame(rows)
    df.to_csv("simopt_comparison.csv", index=False)
    print("\n=== Risultati ===")
    print(df[["algo", "run", "best_disc"]].to_string(index=False))
    print(f"\nTempo totale: {time.time()-t0:.1f}s")

    # grafico comparativo
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        sns.barplot(data=df, x="algo", y="best_disc", errorbar="sd")
        plt.ylabel("Discrepancy ↓")
        plt.title("SimOpt – migliore discrepanza (±sd su run)")
        plt.tight_layout()
        plt.savefig("simopt_algo_compare.png")
    except Exception as e:
        print("Plot non generato:", e)


if __name__ == "__main__":
    main()

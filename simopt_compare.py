"""
SimOpt benchmark con più ottimizzatori Nevergrad
------------------------------------------------
• Policy: PPO (Stable-Baselines3)
• Ambiente: CustomHopper-source / target
• Si ottimizzano le prime 3 masse; la quarta resta invariata.
• Algoritmi testati: DE, PSO, NGOpt8
• Gestione robusta di masse fuori range e warning MuJoCo.
"""

import argparse, random, time, warnings

import gym
import nevergrad as ng
import numpy as np
import pandas as pd
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy

from env.custom_hopper import CustomHopper   # assicurati che il tuo package sia visibile

# ---------- global seed ----------
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
warnings.filterwarnings("ignore", category=UserWarning)  # silenzia Monitor SB3

# ---------- helper RL ----------
def train_policy(env: gym.Env, total_timesteps: int = 5_000) -> PPO:
    model = PPO("MlpPolicy", env, learning_rate=1e-3, gamma=0.99,
                seed=SEED, verbose=0)
    model.learn(total_timesteps=total_timesteps)
    return model


def eval_reward(model: PPO, env_id: str, masses: np.ndarray,
                episodes: int) -> float:
    env = gym.make(env_id)
    env.set_parameters(masses)
    mean, _ = evaluate_policy(model, env,
                              n_eval_episodes=episodes, render=False)
    env.close()
    return mean


# ---------- SimOpt singolo ----------
def simopt_once(optim_cls, budget: int, episodes_eval: int,
                init_sigma: float = 0.5) -> dict:
    # masse di partenza (4 valori)
    env_tmp = gym.make("CustomHopper-source-v0")
    base_masses = env_tmp.get_parameters()
    env_tmp.close()

    # parametrizzazione NG: 3 masse, con bounds sicuri
    def scalar(val):
        return ng.p.Scalar(val).set_bounds(0.1, 20).set_mutation(sigma=init_sigma)

    instrum = ng.p.Dict(m0=scalar(base_masses[0]),
                        m1=scalar(base_masses[1]),
                        m2=scalar(base_masses[2]))
    optim = optim_cls(parametrization=instrum, budget=budget)

    best_disc, best_masses = np.inf, None

    for _ in range(budget):
        cand = optim.ask()
        masses3 = np.array([cand["m0"].value,
                            cand["m1"].value,
                            cand["m2"].value])
        masses_full = base_masses.copy()
        masses_full[:3] = masses3

        try:
            # ---------- train PPO ----------
            env_src = gym.make("CustomHopper-source-v0")
            env_src.set_parameters(masses_full)
            model = train_policy(env_src, total_timesteps=5_000)
            env_src.close()

            # ---------- evaluate ----------
            r_t = eval_reward(model, "CustomHopper-target-v0",
                              masses_full, episodes_eval)
            r_s = eval_reward(model, "CustomHopper-source-v0",
                              masses_full, episodes_eval)
            disc = abs(r_t - r_s)

        except Exception as e:                     # MuJoCo warning → penalità
            disc = 1e9
            print("Iterazione saltata:", e)

        optim.tell(cand, disc)

        if disc < best_disc:
            best_disc, best_masses = disc, masses_full.copy()

    return dict(best_disc=best_disc, best_masses=best_masses.tolist())


# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=1)
    ap.add_argument("--budget", type=int, default=100)
    ap.add_argument("--episodes", type=int, default=10)
    args = ap.parse_args()

    ALGOS = {
        "DE":   ng.optimizers.DE,
        "PSO":  ng.optimizers.PSO,
        "NGOpt8": ng.optimizers.NGOpt8,
    }

    rows, t0 = [], time.time()
    for name, cls in ALGOS.items():
        for run in range(args.runs):
            print(f"[{name}] run {run+1}/{args.runs}")
            res = simopt_once(cls, args.budget, args.episodes)
            res.update(algo=name, run=run)
            rows.append(res)

    df = pd.DataFrame(rows)
    df.to_csv("simopt_comparison.csv", index=False)
    print("\n=== Risultati ===")
    print(df[["algo", "run", "best_disc"]].to_string(index=False))
    print(f"Tempo totale: {time.time()-t0:.1f}s")

    # grafico
    try:
        import matplotlib.pyplot as plt, seaborn as sns
        sns.barplot(data=df, x="algo", y="best_disc", errorbar="sd")
        plt.ylabel("Discrepancy ↓")
        plt.title("SimOpt – best discrepancy (±sd)")
        plt.tight_layout()
        plt.savefig("simopt_algo_compare.png")
        plt.close()
    except Exception as e:
        print("Plot non generato:", e)


if __name__ == "__main__":
    main()

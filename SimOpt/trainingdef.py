import gym
import torch
import random
import numpy as np
import nevergrad as ng
import argparse
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from env.custom_hopper import *
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
from scipy.ndimage import gaussian_filter1d
from scipy.stats import wasserstein_distance
from sklearn.metrics.pairwise import rbf_kernel

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

def discrepancy_score1(real_obs, sim_obs, w1=1.0, w2=0.1, sigma=1.0):
    real_obs, sim_obs = np.array(real_obs), np.array(sim_obs)
    diff = sim_obs - real_obs
    return w1 * np.sum(gaussian_filter1d(np.sum(np.abs(diff), axis=1), sigma=sigma)) + \
           w2 * np.sum(gaussian_filter1d(np.sum(diff**2, axis=1), sigma=sigma))

def discrepancy_score2(real_obs, sim_obs, gamma=0.5):
    X, Y = np.vstack(real_obs), np.vstack(sim_obs)
    return np.mean(rbf_kernel(X, X, gamma=gamma)) + \
           np.mean(rbf_kernel(Y, Y, gamma=gamma)) - \
           2 * np.mean(rbf_kernel(X, Y, gamma=gamma))

def discrepancy_score3(real_obs, sim_obs):
    X, Y = np.vstack(real_obs), np.vstack(sim_obs)
    return np.mean([wasserstein_distance(X[:, d], Y[:, d]) for d in range(X.shape[1])])

def rollout_episodes(env, model, episodes=50):
    collected = []
    for _ in range(episodes):
        done, obs, temp = False, env.reset(), []
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, done, _ = env.step(action)
            temp.append(obs)
        collected.append(np.concatenate(temp))
    return collected

def compute_discrepancy(real_obs, sim_obs, method):
    min_len = min(min(len(o) for o in real_obs), min(len(o) for o in sim_obs))
    real_obs = [o[:min_len] for o in real_obs]
    sim_obs = [o[:min_len] for o in sim_obs]

    if method == "score1":
        return discrepancy_score1(real_obs, sim_obs)
    elif method == "score2":
        return discrepancy_score2(real_obs, sim_obs)
    elif method == "score3":
        return discrepancy_score3(real_obs, sim_obs)
    else:
        raise ValueError("Invalid discrepancy method selected.")

def simopt_loop(mu_vars, discrepancy_method):
    tol = 1e-3
    while all(var[1] > tol for var in mu_vars):
        masses = [np.random.normal(mu[0], mu[1]) for mu in mu_vars]
        env_sim = gym.make('CustomHopper-source-v0')
        env_sim.set_parameters(masses)
        model = PPO("MlpPolicy", env_sim, learning_rate=0.001, gamma=0.99, verbose=0, seed=SEED)
        model.learn(total_timesteps=10000)

        env_real = gym.make('CustomHopper-target-v0')
        env_real.set_parameters(masses)

        real_obs = rollout_episodes(env_real, model)
        sim_obs = rollout_episodes(env_sim, model)

        discrepancy = compute_discrepancy(real_obs, sim_obs, discrepancy_method)
        print(f"Discrepancy ({discrepancy_method}):", discrepancy)

        param = ng.p.Dict(**{
            f"x{i+1}": ng.p.Scalar(init=mu_vars[i][0]).set_mutation(sigma=mu_vars[i][1])
            for i in range(3)
        })
        optimizer = ng.optimizers.CMA(parametrization=param, budget=1300)

        for _ in range(optimizer.budget):
            x = optimizer.ask()
            optimizer.tell(x, discrepancy)

        rec = optimizer.recommend()
        print("Recommended:", rec.value)

        for i, k in enumerate(['x1', 'x2', 'x3']):
            samples = np.append(np.random.normal(mu_vars[i][0], mu_vars[i][1], 300), rec.value[k])
            mu_vars[i][0], mu_vars[i][1] = np.mean(samples), np.var(samples)

        print("Updated mu/var:", mu_vars)

    return mu_vars

def final_training(mu_vars, total_steps):
    masses = [np.random.normal(mu[0], mu[1]) for mu in mu_vars]
    env_train = gym.make('CustomHopper-source-v0')
    env_train.set_parameters(masses)
    model = PPO("MlpPolicy", env_train, learning_rate=0.001, gamma=0.99, verbose=0, seed=SEED)

    env_eval = Monitor(gym.make('CustomHopper-target-v0'))
    rewards = {}
    log = []

    for step in range(1000, total_steps + 1, 1000):
        model.learn(total_timesteps=1000, reset_num_timesteps=False)
        avg, _ = evaluate_policy(model, env_eval, n_eval_episodes=50)
        rewards[step] = [avg]
        log.append(["Source-Target", step, avg])
        print(f"Step {step}, Eval: {avg:.2f}")

    np.save("SimOpt_results.npy", rewards)
    df = pd.DataFrame(log, columns=["Environment", "Timesteps", "Mean Reward"])
    plt.figure(figsize=(12, 8))
    sns.lineplot(data=df, x="Timesteps", y="Mean Reward", hue="Environment", errorbar="sd")
    plt.title("SimOpt Final PPO Performance")
    plt.savefig("SimOpt_Performance.png")
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--discrepancy", choices=["score1", "score2", "score3"], default="score1")
    parser.add_argument("--final_steps", type=int, default=100000, help="Total training steps for final training")
    args = parser.parse_args()

    mu_init = [[3.92699082, 0.5], [2.71433605, 0.5], [5.0893801, 0.5]]
    mu_final = simopt_loop(mu_init, args.discrepancy)
    final_training(mu_final,args.final_steps)

if __name__ == '__main__':
    main()

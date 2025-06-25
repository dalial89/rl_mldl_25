import gym
from pathlib import Path
import torch
import random
import numpy as np
import nevergrad as ng
import argparse
import pandas as pd

from env.custom_hopper import *
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
from scipy.ndimage import gaussian_filter1d
from scipy.stats import wasserstein_distance
from sklearn.metrics.pairwise import rbf_kernel

def make_env(env_id: str, seed: int):
    env = gym.make(env_id)
    env.seed(seed)
    env.action_space.seed(seed)
    return env

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
    env_template = make_env("CustomHopper-source-v0", SEED)
    root_mass = env_template.sim.model.body_mass[1]      
    env_template.close()

    while all(var[1] > tol for var in mu_vars):
        masses3 = [np.random.normal(mu[0], mu[1]) for mu in mu_vars]   # thigh, leg, foot
        masses4 = np.concatenate([[root_mass], masses3])               # prepend torso
        env_sim  = make_env("CustomHopper-source-v0", SEED)
        env_sim.set_parameters(masses4)
        model = PPO("MlpPolicy", env_sim,
                    learning_rate=3e-4, gamma=0.99,
                    verbose=0, seed=SEED, device=args.device)
        model.learn(total_timesteps=10000)

        env_real = make_env("CustomHopper-target-v0", SEED)
        env_real.set_parameters(masses4)

        real_obs = rollout_episodes(env_real, model)
        sim_obs = rollout_episodes(env_sim, model)

        discrepancy = compute_discrepancy(real_obs, sim_obs, discrepancy_method)
        print(f"Discrepancy ({discrepancy_method}):", discrepancy)

        param = ng.p.Dict(**{
            f"x{i+1}": ng.p.Scalar(init=mu_vars[i][0]).set_mutation(sigma=mu_vars[i][1])
            for i in range(3)
        })
        optimizer = ng.optimizers.CMA(parametrization=param,
                              budget=1300,
                              random_state=np.random.RandomState(SEED))

        for _ in range(optimizer.budget):
            x = optimizer.ask()
            optimizer.tell(x, discrepancy)

        rec = optimizer.recommend()
        print("Recommended:", rec.value)

        for i, k in enumerate(['x1', 'x2', 'x3']):
            samples = np.append(np.random.normal(mu_vars[i][0], mu_vars[i][1], 300), rec.value[k])
            mu_vars[i][0], mu_vars[i][1] = np.mean(samples), np.var(samples)

        print("Updated mu/var:", mu_vars)

    return mu_vars, root_mass

def final_training(mu_vars, root_mass, total_steps):
    masses3 = [np.random.normal(mu[0], mu[1]) for mu in mu_vars]
    masses4 = np.concatenate([[root_mass], masses3])
    env_train = make_env("CustomHopper-source-v0", SEED)
    env_train.set_parameters(masses4)
    env_train = Monitor(env_train)
    
    model = PPO("MlpPolicy", env_train,
                learning_rate=3e-4, gamma=0.99,
                verbose=1, seed=SEED, device=args.device)

    env_eval  = Monitor(make_env("CustomHopper-target-v0", SEED))
    log = []

    steps_done = 0
    mean_training = np.nan
    while steps_done < total_steps:
        steps_to_do = min(1000, total_steps - steps_done)
        model.learn(total_timesteps=steps_to_do, reset_num_timesteps=False)
        steps_done += steps_to_do
        train_rewards = env_train.get_episode_rewards()
        if len(train_rewards) >= 10:
            mean_training = np.mean(train_rewards[-10:])
            print(f"Mean training reward (last 10 episodes): {mean_training:.2f}")
        avg_eval, _ = evaluate_policy(model, env_eval, n_eval_episodes=50)
        #print(f"[{step}] Evaluation on target: {avg_eval:.2f}")
        log.append(["Train (source)", steps_done, mean_training])
        log.append(["Eval (target)", steps_done, avg_eval])

    env_type = ("source" if "source" in env_train.unwrapped.spec.id.lower()
                         else "target")
    tag = f"ppo_tuned_{env_type}_seed_{SEED}_simopt_{args.discrepancy}"

    # repos
    base_dir = Path(__file__).resolve().parents[1]   
    w_dir   = base_dir / "models_weights"
    d_dir   = base_dir / "models_data"
    w_dir.mkdir(exist_ok=True, parents=True)
    d_dir.mkdir(exist_ok=True, parents=True)

    # 1) save weights
    model.save(str(w_dir / f"{tag}.zip"))

    # 2) save csv with returns
    df = pd.DataFrame(log,
                      columns=["Environment", "Timesteps", "Mean Reward"])
    df.to_csv(d_dir / f"{tag}.csv", index=False)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--discrepancy", choices=["score1", "score2", "score3"], default="score1")
    parser.add_argument("--final_steps", type=int, default=100000, help="Total training steps for final training")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (forwarded by main.py)")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu", help="Torch device")
    global args          
    args = parser.parse_args()

    global SEED
    SEED = args.seed
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    mu_init = [[3.92699082, 0.5], [2.71433605, 0.5], [5.0893801, 0.5]]
    mu_final, root_mass = simopt_loop(mu_init, args.discrepancy)
    final_training(mu_final,root_mass, args.final_steps)

if __name__ == '__main__':
    main()
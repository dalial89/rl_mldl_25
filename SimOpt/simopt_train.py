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

# Create a gym environment
def make_env(env_id: str, seed: int):
    env = gym.make(env_id)
    env.seed(seed)
    env.action_space.seed(seed)
    return env

# Discrepancy Score 1: Smoothed sum of absolute and squared differences
# Parameters w1, w2 weight the importance of L1 and L2 components, sigma controls Gaussian smoothing
def discrepancy_score1(real_obs, sim_obs, w1=1.0, w2=0.1, sigma=1.0):
    real_obs, sim_obs = np.array(real_obs), np.array(sim_obs)
    diff = sim_obs - real_obs
    return w1 * np.sum(gaussian_filter1d(np.sum(np.abs(diff), axis=1), sigma=sigma)) + \
           w2 * np.sum(gaussian_filter1d(np.sum(diff**2, axis=1), sigma=sigma))

# Discrepancy Score 2: Kernel-based Maximum Mean Discrepancy
def discrepancy_score2(real_obs, sim_obs, gamma=0.5):
    X, Y = np.vstack(real_obs), np.vstack(sim_obs)
    return np.mean(rbf_kernel(X, X, gamma=gamma)) + \
           np.mean(rbf_kernel(Y, Y, gamma=gamma)) - \
           2 * np.mean(rbf_kernel(X, Y, gamma=gamma))

# Discrepancy Score 3: Mean Wasserstein distance across all observation dimensions
def discrepancy_score3(real_obs, sim_obs):
    X, Y = np.vstack(real_obs), np.vstack(sim_obs)
    return np.mean([wasserstein_distance(X[:, d], Y[:, d]) for d in range(X.shape[1])])

# Collect full observation trajectories from the environment
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

# Align trajectory lengths and compute the selected discrepancy
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

# SimOpt optimization loop
'''
def simopt_loop(mu_vars, discrepancy_method, optimizer_name):
    tol = 1e-3
    # Create environment and retrieve the root (torso) mass
    env_template = make_env("CustomHopper-source-v0", SEED)
    root_mass = env_template.sim.model.body_mass[1]      
    env_template.close()

    # Repeat until all variances in the mass distribution are below the threshold
    while all(var[1] > tol for var in mu_vars):
        # 1) Sample a new set of dynamic masses (thigh, leg, foot) from current mean and variance
        masses3 = [np.random.normal(mu[0], mu[1]) for mu in mu_vars]   # thigh, leg, foot
        masses4 = np.concatenate([[root_mass], masses3])               # prepend torso

        # 2) Create a simulated environment with the new body parameters and train PPO agent
        env_sim  = make_env("CustomHopper-source-v0", SEED)
        env_sim.set_parameters(masses3)

        model = PPO("MlpPolicy", env_sim,
                    learning_rate=3e-4, gamma=0.99,
                    verbose=0, seed=SEED, device=args.device)
        model.learn(total_timesteps=10000)

        # 3) Create the "real" environment with the same parameters (sim-to-real simulation)
        env_real = make_env("CustomHopper-target-v0", SEED)
        env_real.set_parameters(masses3)

        # Run rollouts in both simulated and real environments using the trained model
        real_obs = rollout_episodes(env_real, model)
        sim_obs = rollout_episodes(env_sim, model)

        # 4) Compute discrepancy between trajectories from simulation and real environments
        discrepancy = compute_discrepancy(real_obs, sim_obs, discrepancy_method)
        print(f"Discrepancy ({discrepancy_method}):", discrepancy)

        # 5) Define parameter space for the optimization
        param = ng.p.Dict(**{
            f"x{i+1}": ng.p.Scalar(init=mu_vars[i][0]).set_mutation(sigma=mu_vars[i][1])
            for i in range(3)
        })
        # Select optimizer based on args.optimizer
        if optimizer_name == "cma":
            optimizer = ng.optimizers.CMA(parametrization=param, budget=1300)
        elif optimizer_name == "pso":
            optimizer = ng.optimizers.PSO(parametrization=param, budget=1300)
        elif optimizer_name == "de":
            optimizer = ng.optimizers.DE(parametrization=param, budget=1300)
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")
            
        # set the seed on the underlying Instrumentation RNG
        optimizer.parametrization.random_state.seed(SEED)

        # Feed the optimizer with current discrepancy score
        for _ in range(optimizer.budget):
            x = optimizer.ask()
            optimizer.tell(x, discrepancy)

        # 6) Retrieve best suggestion from optimizer
        rec = optimizer.recommend()
        print("Recommended:", rec.value)

        # Update mu/var estimates by fitting new distribution to sampled values + recommended ones
        for i, k in enumerate(['x1', 'x2', 'x3']):
            samples = np.append(np.random.normal(mu_vars[i][0], mu_vars[i][1], 300), rec.value[k])
            mu_vars[i][0], mu_vars[i][1] = np.mean(samples), np.var(samples)

        print("Updated mu/var:", mu_vars)

    return mu_vars, root_mass #optimized distribution parameters and torso mass
    '''
def simopt_loop(mu_vars, discrepancy_method, optimizer_name):
    tol = 1e-3

    # Get root mass from the base environment (assumed fixed)
    env_template = make_env("CustomHopper-source-v0", SEED)
    root_mass = env_template.sim.model.body_mass[1]
    env_template.close()

    iteration = 0

    while all(var[1] > tol for var in mu_vars):
        print(f"\n--- Iteration {iteration} ---")
        
        # Define Nevergrad parameter space based on current mu/var
        param = ng.p.Dict(**{
            f"x{i+1}": ng.p.Scalar(init=mu_vars[i][0]).set_mutation(sigma=mu_vars[i][1])
            for i in range(3)
        })

        # Select optimizer
        if optimizer_name == "cma":
            optimizer = ng.optimizers.CMA(parametrization=param, budget=5)
        elif optimizer_name == "pso":
            optimizer = ng.optimizers.PSO(parametrization=param, budget=5)
        elif optimizer_name == "de":
            optimizer = ng.optimizers.DE(parametrization=param, budget=5)
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")

        optimizer.parametrization.random_state.seed(SEED)

        # Evaluate each candidate sample proposed by the optimizer
        for _ in range(optimizer.budget):
            x = optimizer.ask()
            masses3 = [float(x[f"x{i+1}"]) for i in range(3)]

            # Train PPO on sim env with current masses
            env_sim = make_env("CustomHopper-source-v0", SEED)
            env_sim.set_parameters(masses3)

            model = PPO("MlpPolicy", env_sim,
                        learning_rate=3e-4, gamma=0.99,
                        verbose=0, seed=SEED, device=args.device)
            model.learn(total_timesteps=10000)

            # Evaluate discrepancy between sim and real environments
            env_real = make_env("CustomHopper-target-v0", SEED)
            env_real.set_parameters(masses3)

            real_obs = rollout_episodes(env_real, model)
            sim_obs  = rollout_episodes(env_sim, model)
            discrepancy = compute_discrepancy(real_obs, sim_obs, discrepancy_method)

            print(f"Candidate: {masses3}, Discrepancy: {discrepancy:.4f}")
            optimizer.tell(x, discrepancy)

        # Get best mass values from optimizer
        rec = optimizer.recommend()
        print("Recommended masses:", rec.value)

        # Update distribution mu/var with samples + best candidate
        for i, k in enumerate(['x1', 'x2', 'x3']):
            samples = np.append(
                np.random.normal(mu_vars[i][0], mu_vars[i][1], 300),
                rec.value[k]
            )
            mu_vars[i][0], mu_vars[i][1] = np.mean(samples), np.var(samples)

        print("Updated mu/var:", mu_vars)
        iteration += 1

    return mu_vars, root_mass


# Final training phase using optimized mass parameters
def final_training(mu_vars, root_mass, total_steps):
    # Sample final body masses from optimized distribution
    masses3 = [np.random.normal(mu[0], mu[1]) for mu in mu_vars]
    masses4 = np.concatenate([[root_mass], masses3])

    # Set up training environment with final parameters
    env_train = make_env("CustomHopper-source-v0", SEED)
    env_train.set_parameters(masses3)
    env_train = Monitor(env_train)

    # Initialize PPO model
    model = PPO("MlpPolicy", env_train,
                learning_rate=3e-4, gamma=0.99,
                verbose=1, seed=SEED, device=args.device)

    # Train 
    model.learn(total_timesteps=total_steps)

    # Evaluate final policy 
    env_eval  = Monitor(make_env("CustomHopper-target-v0", SEED))
    avg_eval, _ = evaluate_policy(model, env_eval, n_eval_episodes=50)
    train_rewards = env_train.get_episode_rewards()
    mean_training = np.mean(train_rewards[-10:]) if len(train_rewards) >= 10 else np.mean(train_rewards)
    log = [
        ["Train (source)", total_steps, mean_training],
        ["Eval (target)", total_steps, avg_eval]
    ]


    # Prepare logging
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
    parser.add_argument("--optimizer", choices=["cma", "pso", "de"], default="cma", help="Optimization algorithm to use")
    global args          
    args = parser.parse_args()
    global SEED
    SEED = args.seed
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    mu_init = [[3.92699082, 0.5], [2.71433605, 0.5], [5.0893801, 0.5]]
    mu_final, root_mass = simopt_loop(mu_init, args.discrepancy, args.optimizer)
    final_training(mu_final,root_mass, args.final_steps)

if __name__ == '__main__':
    main()
    main()

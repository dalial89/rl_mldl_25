import gym
import pandas as pd
import numpy as np
import time
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from env.custom_hopper import CustomHopper

pd.set_option("display.max_columns", None)


def evaluate_setup(model_path, env_name, seed, udr, setup_name, episodes=100):
    set_seed(seed)
    
    model = PPO.load(model_path)
    MAX_STEPS = 1000  # Maximum steps per episode
    
    env = make_env(env_name, udr=udr, max_steps=max_steps)

    episode_rewards = []
    episode_lengths = []

    TIMEOUT = 10      # Maximum seconds per episode

    for ep in range(episodes):
        obs = env.reset()
        done = False
        total_reward = 0.0
        steps = 0
        start_time = time.time()

        while not done and steps < MAX_STEPS:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)

            if not np.isfinite(reward) or not np.all(np.isfinite(obs)):
                print(f"[WARNING] Episode {ep+1}: Non-finite reward or obs — breaking.")
                break

            total_reward += reward
            steps += 1

            if time.time() - start_time > TIMEOUT:
                print(f"[WARNING] Episode {ep+1}: Timeout after {TIMEOUT}s — breaking.")
                break

        episode_rewards.append(total_reward)
        episode_lengths.append(steps)
        print(f"[DEBUG] Episode {ep+1} — Reward: {total_reward:.2f} | Steps: {steps}")

    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)

    return {
        'setup': setup_name,
        'env': env_name,
        'seed': seed,
        'udr': udr,
        'mean': mean_reward,
        'std': std_reward,
        'episodes': episodes
    }


def main():
    results = []

    # Configurations to evaluate
    configs = [
        ('Simopt_ppo_policy_final', 'CustomHopper-source-v0', 42, True, 'source→source'),
        ('Simopt_ppo_policy_final', 'CustomHopper-target-v0', 42, True, 'source→target'),
        ('Simopt_ppo_policy_final', 'CustomHopper-target-v0', 42, True, 'target→target')
    ]

    for model_path, env_name, seed, udr, setup_name in configs:
        result = evaluate_setup(model_path, env_name, seed, udr, setup_name)
        results.append(result)

    # Create and save result table
    df_results = pd.DataFrame(results)
    df_results.to_csv('results_table.csv', index=False)
    print("\nFinal Results:\n", df_results)


if __name__ == '__main__':
    main()

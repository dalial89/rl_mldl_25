import argparse
import random
import sys

import gym
import matplotlib
import nevergrad as ng
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from collections import defaultdict
from matplotlib import pyplot as plt
from scipy.ndimage import gaussian_filter1d
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor

from env.custom_hopper import *

matplotlib.use('Agg')

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

def setup_environment(domain, parameters):
	env = CustomHopper(domain, mass_dist_params=parameters)
	env.seed(SEED)
	env.action_space.seed(SEED)
	return env

def train_agent(env, steps=10000):
	model = PPO("MlpPolicy", env, learning_rate=0.001, gamma = 0.99 , verbose=0, seed=SEED)
	model.learn(total_timesteps=steps)
	model.save("Simopt_ppo_policy")
	return model

def evaluate_policy_on_env(env, model, n_episodes=50):
	cumulative_reward = 0
	collected_obs = []
	for _ in range(n_episodes):
		obs_list = []
		obs = env.reset()
		done = False
		while not done:
			action, _ = model.predict(obs, deterministic=True)
			obs, reward, done, _ = env.step(action)
			obs_list.append(obs)
			cumulative_reward += reward
		collected_obs.append(np.concatenate(obs_list))
	return cumulative_reward / n_episodes, collected_obs

def discrepancy_score(real_obs, sim_obs,w1=1.0, w2=0.1, sigma=1.0):
	real_obs = np.array(real_obs)
	sim_obs = np.array(sim_obs)
	diff = sim_obs - real_obs
	l1 = gaussian_filter1d(np.sum(np.abs(diff), axis=1), sigma=sigma)
	l2 = gaussian_filter1d(l2_norm = np.sum(diff ** 2, axis=1), sigma=sigma)
	discrepancy = w1 * np.sum(l1) + w2 * np.sum(l2)
	return discrepancy

def update_distribution(mu_vars, recommendation):
	updated = []
	for i, key in enumerate(['x1', 'x2', 'x3']):
		samples = np.random.normal(mu_vars[i][0], mu_vars[i][1], 300)
		new_val = recommendation[key].value
		samples = np.append(samples, new_val)
		new_mean = np.mean(samples)
		new_var = np.var(samples)
		updated.append([new_mean, new_var])
	return updated

def optimize_parameters(mu_vars):
	params = ng.p.Dict(x1=ng.p.Scalar(), x2=ng.p.Scalar(), x3=ng.p.Scalar())
	for i, k in enumerate(['x1', 'x2', 'x3']):
		params[k].value = mu_vars[i][0]
		params[k].set_mutation(sigma=mu_vars[i][1])
		params[k].mutate()
	opt = ng.optimizers.CMA(parametrization=params, budget=1300)
	return opt
 
def main():	
	mu_vars = [[3.92699082, 0.5], [2.71433605, 0.5], [5.0893801, 0.5]]
	tol = 1e-3 #will be used as stopping criterion for the following cycle 

	while all(v[1] > tol for v in mu_vars):
		sim_env = gym.make('CustomHopper-source-v0')
		masses = sim_env.get_parameters()
		for i in range(1, 4):
			masses[i] = np.random.normal(mu_vars[i - 1][0], mu_vars[i - 1][1], 1)
		sim_env.set_parameters(masses[1:4])
		train_agent(sim_env)
		policy = PPO.load("Simopt_ppo_policy")

		real_env = gym.make('CustomHopper-target-v0')
		real_env.set_parameters(masses[1:])
		real_avg, real_obs = evaluate_policy_on_env(real_env, policy)
		print(f"Average Return REAL: {real_avg:.2f}")
		
		sim_env = gym.make('CustomHopper-source-v0')
		sim_env.set_parameters(masses[1:])
		sim_avg, sim_obs = evaluate_policy_on_env(sim_env, policy)
		print(f"Average Return SIM: {sim_avg:.2f}")	
		
		#DISCREPANCY
		# Calculate the discrepancy between simulation and reality using the discrepancy function
		min_length = min(min(len(obs) for obs in real_obs), min(len(obs) for obs in sim_obs))
		real_obs = [obs[:min_length] for obs in real_obs]
		sim_obs = [obs[:min_length] for obs in sim_obs]
		discrepancy = discrepancy_score(real_obs, sim_obs)
		print("Discrepancy: ", discrepancy)
		
		#OPTIMIZE AND UPDATE 		
		optimizer = optimize_parameters(mu_vars)
		for _ in range(optimizer.budget):
			candidate = optimizer.ask()
			optimizer.tell(candidate, discrepancy)
		recommendation = optimizer.recommend()
		print("Best candidate:", recommendation.value)
		
		mu_vars = update_distribution(mu_vars, recommendation)
		print("Updated distributions:", mu_vars)

#TRAIN THE DEFINITIVE MODEL
	n_eval_episodes = 50
	eval_interval = 1000 
	total_timesteps = 50000
	
	sim_env = Monitor(gym.make('CustomHopper-source-v0'))
	masses = sim.get_parameters()
	for i in range(3):
		masses[i + 1] = np.random.normal(mu_vars[i][0], mu_vars[i][1], 1)
	sim_env.set_parameters(masses[1:])

	model = PPO("MlpPolicy", sim_env, learning_rate=0.001, gamma = 0.99 , verbose=0, seed=42)

	source_rewards = {i: [] for i in range(eval_interval, total_timesteps + 1, eval_interval)}
	test_env = Monitor(gym.make('CustomHopper-target-v0'))

	all_episode_rewards = []
	reward_log = []
				
    	#Evaluate the final model
	for step in range(eval_interval, total_timesteps + 1, eval_interval):
		model.learn(total_timesteps= eval_interval, reset_num_timesteps=False)
		mean_reward, _ = evaluate_policy(model, test_env, n_eval_episodes=50, render=False)
		source_rewards[step].append(mean_reward)
		episode_rewards = sim_env.get_episode_rewards()
		if episode_rewards:
			reward = episode_rewards[-1]
			all_episode_rewards.append(reward)
			running_var = 0.0 if len(all_episode_rewards) == 1 else np.var(all_episode_rewards)
			reward_log.append((step, reward, running_var))
		print(f"Step: {step}, Mean Reward: {mean_reward:.2f}")
		
	model.save("Simopt_ppo_policy_final")


    #save the results
	df1 = pd.DataFrame([(k, v[0]) for k, v in source_rewards.items()], columns=["Timesteps", "Mean Reward"])
	df1['Environment'] = 'Source-Target'
	df_log = pd.DataFrame(reward_log, columns=["step", "reward", "running_variance"])
	df_log.to_csv("training_step_reward_variance.csv", index=False)
	np.save('SimOpt_results.npy', source_rewards)

    #Plot the results
	plt.figure(figsize=(12, 8))
	sns.lineplot(x='Timesteps', y='Mean Reward', hue='Environment', data=df1, errorbar='sd')
	plt.title('SimOpt Performance')
	plt.ylabel('Mean Reward')
	plt.xlabel('Training Timesteps')
	plt.savefig('SimOpt_Performance.png')
	plt.close()




if __name__ == '__main__':
	main()	



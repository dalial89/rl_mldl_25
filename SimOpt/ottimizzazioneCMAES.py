import torch
import gym
import argparse
import nevergrad as ng
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import sys
from env.custom_hopper import *
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from scipy.ndimage import gaussian_filter1d

matplotlib.use('Agg')

def make_env(domain, mass_dist_params):
    return CustomHopper(domain, mass_dist_params=mass_dist_params)

def train_policy(env, total_timesteps=10000):
    model = PPO("MlpPolicy", env, learning_rate=0.001, gamma = 0.99 , verbose=0) #train the model
    model.learn(total_timesteps=total_timesteps)
    model.save("Simopt_ppo_policy")
    return model

def compute_discrepancy(real_obs, sim_obs): #computing of discrepancy

	w1 = 1.0
	w2 = 0.1
	sigma = 1.0
	real_obs = np.array(real_obs)
	sim_obs = np.array(sim_obs)
	W = np.ones(real_obs.shape)
	assert sim_obs.shape == real_obs.shape, "Observation shapes do not match"
    
    # Compute the weighted differences
	diff = W * (sim_obs - real_obs)
    
    # Compute 1 and 2 norms
	l1_norm = np.sum(np.abs(diff), axis=1)
	l2_norm = np.sum(diff ** 2, axis=1)
    
    # Apply Gaussian filter to smooth the norms
	l1_norm_smoothed = gaussian_filter1d(l1_norm, sigma=sigma)
	l2_norm_smoothed = gaussian_filter1d(l2_norm, sigma=sigma)
    
    # Compute the weighted sum of norms
	discrepancy = w1 * np.sum(l1_norm_smoothed) + w2 * np.sum(l2_norm_smoothed)

	return discrepancy
 
def main():
	# Setting the tolerance for the variance
	tol = 1e-3
	
    	#definition of mean and variances
	mu_std1 = [3.92699082, 0.5]
	mu_std2 = [2.71433605, 0.5]
	mu_std3 = [5.0893801, 0.5]
	
	# Iterate till each variance is less than the tolerance
	while mu_std1[1]>tol and mu_std2[1]>tol and mu_std3[1]>tol:
		
		# Train the policy simulation(p_phi_i)
		sim_env = gym.make('CustomHopper-source-v0')

		masses = sim_env.get_parameters()
		masses[1] = np.random.normal(mu_std1[0], mu_std1[1], 1)
		masses[2] = np.random.normal(mu_std2[0], mu_std2[1], 1)
		masses[3] = np.random.normal(mu_std3[0], mu_std3[1], 1)

		sim_env.set_parameters(masses[1:4])
		
		model = train_policy(sim_env)
		
		#NOW I HAVE THE POLICY  PI_theta_p
		
		# Load the trained PPO model
		model = PPO.load("Simopt_ppo_policy")
		
		#REAL ROLLOUT - same parameters
		real = gym.make('CustomHopper-target-v0')
		real.set_parameters(masses[1:])
		
		tot_rw_real = 0
		episodes = 50
		obs_real = []
		
		# Test the policy in real
		for episode in range(episodes):
			done = False
			train_reward = 0
			obs = real.reset()  
			test_reward = 0
			
			episode_obs = []

			while not done:  
				action, _states = model.predict(obs, deterministic=True)
				obs, reward, done, info = real.step(action)
				# Collect the observation of each step
				episode_obs.append(obs)

				test_reward += reward
			# Collect the observations of each episode	
			obs_real.append(np.concatenate(episode_obs))
			
			tot_rw_real +=test_reward
			
		print(f"Average Return REAL: {tot_rw_real/episodes}")
			
		#SAMPLE + SIM ROLLOUT
		sim = gym.make('CustomHopper-source-v0')
		# Sample the values of the masses from the normal distributions with mean and variance as above
		masses = sim.get_parameters()
		masses[1] = np.random.normal(mu_std1[0], mu_std1[1], 1)
		masses[2] = np.random.normal(mu_std2[0], mu_std2[1], 1)
		masses[3] = np.random.normal(mu_std3[0], mu_std3[1], 1)
		sim.set_parameters(masses[1:])
		
		tot_rw_sim = 0
		obs_sim = []
			
		# Test the policy in simulation
		for episode in range(episodes):
			done = False
			train_reward = 0
			obs = sim.reset()  
			test_reward = 0
			k = 0
			episode_obs = []
			while not done:  
				action, _states = model.predict(obs, deterministic=True)
				obs, reward, done, info = sim.step(action)
				# Collect the observation of each step
				episode_obs.append(obs)

				test_reward += reward
			# Collect the observations of each episode
			obs_sim.append(np.concatenate(episode_obs))	
			
			tot_rw_sim +=test_reward
		print(f"Average Return SIM: {tot_rw_sim/episodes}")
		
		#DISCREPANCY
		# Calculate the discrepancy between simulation and reality using the discrepancy function
		min_length = min(min(len(obs) for obs in obs_real), min(len(obs) for obs in obs_sim))
		obs_real = [obs[:min_length] for obs in obs_real]
		obs_sim = [obs[:min_length] for obs in obs_sim]
		#discrepancy = np.linalg.norm(np.array(obs_real)-np.array(obs_sim))
		discrepancy = compute_discrepancy(obs_real, obs_sim)
		print("Discrepancy: ", discrepancy)
		
		#UPDATE DISTRIBUTION
		# Setting the parameters for the optimizer
		# x1, x2, x3 will represent the mean values of the distributions of the masses
		# Initialize them as scalars
		param = ng.p.Dict(
			x1 = ng.p.Scalar(),
			x2 = ng.p.Scalar(),
			x3 = ng.p.Scalar()
		)		
		
		# Set their value
		param["x1"].value = mu_std1[0]
		param["x2"].value = mu_std2[0]
		param["x3"].value = mu_std3[0]
		
		# Set their variation using the variance values of the distributions of the masses
		param["x1"].set_mutation(sigma = mu_std1[1])
		param["x2"].set_mutation(sigma = mu_std2[1])
		param["x3"].set_mutation(sigma = mu_std3[1])
		
		# Take a value for each x1, x2, x3 in the interval [mean - variance, mean + variance]
		param["x1"].mutate()
		param["x2"].mutate()
		param["x3"].mutate()
			
		# Initialize the optimizer, 1300 evaulations Covariance Matrix Adaptation Evolution Strategy 
		optim = ng.optimizers.CMA(parametrization = param, budget=1300)
		
		# Use the retrieved value x1, x2, x3 to start the iterations of the optimizers
		for _ in range(optim.budget):
			x = optim.ask()
			optim.tell(x, discrepancy)
			
		# Use "recommend" to obtain the candidates x1, x2, x3 values with minimal loss (i.e. that minimize the discrepancy)
		recommendation = optim.recommend()
		print(recommendation.value)
		
		# In order to update the distributions, select 300 samples from the actual distributions
		samples1 = np.random.normal(mu_std1[0], mu_std1[1], 300)
		samples2 = np.random.normal(mu_std2[0], mu_std2[1], 300)
		samples3 = np.random.normal(mu_std3[0], mu_std3[1], 300)
		
		# Add to these samples the new recommended values in order to modify the distributions
		# in the direction of the minimization of the discrepancy
		samples1 = np.append(samples1, np.array(recommendation['x1'].value).reshape(1,), axis=0)
		samples2 = np.append(samples2, np.array(recommendation['x2'].value).reshape(1,), axis=0)
		samples3 = np.append(samples3, np.array(recommendation['x3'].value).reshape(1,), axis=0)
		
		# Update the mean values of the distributions
		mu_std1[0] = np.mean(samples1)
		mu_std2[0] = np.mean(samples2)
		mu_std3[0] = np.mean(samples3)
		
		# Update the variance values of the distributions
		mu_std1[1] = np.var(samples1)
		mu_std2[1] = np.var(samples2)
		mu_std3[1] = np.var(samples3)
		
		print(f"Updated distributions: {mu_std1}, {mu_std2}, {mu_std3}")

	#TRAIN THE DEFINITIVE MODEL
	n_policies = 3
	n_eval_episodes = 50
	eval_interval = 1000 
	total_timesteps = 100000
	source_rewards = {i: [] for i in range(eval_interval, total_timesteps + 1, eval_interval)}
	test_env = gym.make('CustomHopper-target-v0')
	test_env = Monitor(test_env)

	for _ in range(n_policies):
		sim_env = gym.make('CustomHopper-source-v0')
		masses = sim.get_parameters()
		masses[1] = np.random.normal(mu_std1[0], mu_std1[1], 1)
		masses[2] = np.random.normal(mu_std2[0], mu_std2[1], 1)
		masses[3] = np.random.normal(mu_std3[0], mu_std3[1], 1)
		sim_env.set_parameters(masses[1:])
		model = PPO("MlpPolicy", sim_env, learning_rate=0.001, gamma = 0.99 , verbose=0) #train the model	

    		# Evaluate the final model
		for step in range(eval_interval, total_timesteps + 1, eval_interval):
			model.learn(total_timesteps= eval_interval, reset_num_timesteps=False)
			mean_reward, _ = evaluate_policy(model, test_env, n_eval_episodes=50, render=False)
			source_rewards[step].append(mean_reward)
		
		model.save("Simopt_ppo_policy_final")

    # Prepare data for plot
	np.save('SimOpt_results.npy', source_rewards)
	plot_data = []
	for step in source_rewards:
		for reward in source_rewards[step]:
			plot_data.append(['Source-Target', step, reward])

	df = pd.DataFrame(plot_data, columns=['Environment', 'Timesteps', 'Mean Reward'])

    # Plot the results
	plt.figure(figsize=(12, 8))
	sns.lineplot(x='Timesteps', y='Mean Reward', hue='Environment', data=df, errorbar='sd')
	plt.title('SimOpt Performance')
	plt.ylabel('Mean Reward')
	plt.xlabel('Training Timesteps')
	plt.savefig("graficoSimOpt.png")



if __name__ == '__main__':
	main()	

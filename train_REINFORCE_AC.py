"""Train an RL agent on the OpenAI Gym Hopper environment using
    REINFORCE and Actor-critic algorithms
"""
import sys

import numpy as np
import importlib
import argparse

import torch
import gym
import random

from env.custom_hopper import *  

def import_agent_module(agent_name: str):
    """
    Dynamically imports agents
    """
    try:
        module = importlib.import_module(f"agentsandpolicies.{agent_name}.{agent_name}")
    except ModuleNotFoundError as exc:
        print(f"[ERROR] agent {agent_name} not found!", file=sys.stderr)
        raise exc

    # check if you have policy and agent classes
    if not all(hasattr(module, cls) for cls in ("Policy", "Agent")):
        raise AttributeError(
            f"[ERROR] {agent_name} should contain 'Policy' and 'Agent' classes!"
        )
    return module.Policy, module.Agent

def run_train(
    agent_name: str,
    n_episodes: int,
    device: str,
    baseline: bool,
    eps: float,
    seed: int
):
    # --- set all seeds -----------------------------------------------------
    # 1) Python
    random.seed(seed)
    # 2) NumPy
    np.random.seed(seed)
    # 3) PyTorch
    torch.manual_seed(seed)
    if device.startswith("cuda"):
        torch.cuda.manual_seed_all(seed)
    # 4) Gym
    try:
        env.seed(seed)
    except AttributeError:
        env.reset(seed=seed)


    # --- import environment ------------------------------------------------
    env = gym.make("CustomHopper-source-v0")
    # env = gym.make('CustomHopper-target-v0')
    
    print("Action space: ", env.action_space)
    print("State space:  ", env.observation_space)
    print("Dynamics parameters:", env.get_parameters())

    # --- import agent ------------------------------------------------------
    PolicyClass, AgentClass = import_agent_module(agent_name)

    obs_dim  = env.observation_space.shape[-1]
    act_dim  = env.action_space.shape[-1]

    policy = PolicyClass(obs_dim, act_dim)
    agent = AgentClass(policy,
                   device=device,
                   baseline=baseline,
                   eps=eps)

    # --- training loop -----------------------------------------------------
    episode_rewards = []
    for episode in range(n_episodes):
        state = env.reset()        
        done, reward_tot = False, 0.0

        while not done:
            action, action_prob = agent.get_action(state)
            prev_state = state

            state, reward, done, info = env.step(action.detach().cpu().numpy())

            if agent_name.lower() == "reinforce":
                agent.store_outcome(action_prob, reward)                         
            else:  
                agent.store_outcome(prev_state, state, action_prob, reward, done)
            reward_tot += reward

        agent.update_policy()
        episode_rewards.append(reward_tot)

        if (episode + 1) % 100 == 0:
            print(f"Episode {episode + 1}: return = {reward_tot:.2f}")

    # --- save -------------------------------------------------------------
    torch.save(agent.policy.state_dict(), f"models_weights/{agent_name}_baseline_{baseline}_eps_{eps}_model.mdl")

    episodes = np.arange(1, len(episode_rewards) + 1)
    rewards  = np.array(episode_rewards)

    data = np.column_stack((episodes, rewards))
    np.savetxt(f"models_data/{agent_name}_baseline_{baseline}_eps_{eps}_returns.csv",
           data,
           delimiter=",",
           header="episode,return",
           comments="")
    
if __name__ == "__main__":


    p = argparse.ArgumentParser(
        description="Train REINFORCE/ActorCritic on Hopper"
    )
    p.add_argument("--seed", type=int, required=True,
               help="Random seed for reproducibility")
    p.add_argument("--agent",    required=True,
                   choices=["REINFORCE","ActorCritic"],
                   help="Which agent to train")
    p.add_argument("--episodes", type=int, required=True,
                   help="Number of training episodes")
    p.add_argument("--device",   choices=["cpu","cuda"],
                   default="cpu",
                   help="Device for training")
    p.add_argument("--baseline", action="store_true",
                   help="Use mean–std baseline")
    p.add_argument("--eps",      type=float,
                   default=1e-8,
                   help="Epsilon for baseline")
    args = p.parse_args()

    run_train(
        agent_name = args.agent,
        n_episodes = args.episodes,
        device     = args.device,
        baseline   = args.baseline,
        eps        = args.eps
    )


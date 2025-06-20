"""Train an RL agent on the OpenAI Gym Hopper environment using
    REINFORCE and Actor-critic algorithms
"""
import os
BASE_DIR = os.path.dirname(os.path.realpath(__file__))
import sys

import numpy as np
import importlib
import argparse

import torch
import gym
import random

from env.custom_hopper import *  

def import_agent_module(agent_name: str):
    mapping = {
        "REINFORCE":      "REINFORCE.REINFORCE",
        "REINFORCE_BAVG": "REINFORCE.REINFORCE_baseline_avg",
        "REINFORCE_BVAL": "REINFORCE.REINFORCE_baseline_value_net",
        "ActorCritic":    "ActorCritic.ActorCritic",
    }
    pkg = mapping.get(agent_name)
    if pkg is None:
        raise ValueError(f"[ERROR] unknown agent '{agent_name}'")

    try:
        module = importlib.import_module(f"agentsandpolicies.{pkg}")
    except ModuleNotFoundError as exc:
        print(f"[ERROR] module '{pkg}' not found!", file=sys.stderr)
        raise

    if not all(hasattr(module, cls) for cls in ("Policy", "Agent")):
        raise AttributeError(f"[ERROR] '{pkg}' must define Policy and Agent")

    return module.Policy, module.Agent


def run_train(
    agent_name: str,
    n_episodes: int,
    device: str,
    seed: int
):
    # --- import environment ------------------------------------------------
    env = gym.make("CustomHopper-source-v0")
    # env = gym.make('CustomHopper-target-v0')

    print("Action space: ", env.action_space)
    print("State space:  ", env.observation_space)
    print("Dynamics parameters:", env.get_parameters())

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

    # --- import agent ------------------------------------------------------
    PolicyClass, AgentClass = import_agent_module(agent_name)

    obs_dim  = env.observation_space.shape[-1]
    act_dim  = env.action_space.shape[-1]

    policy = PolicyClass(obs_dim, act_dim)
    agent = AgentClass(policy, device=device)


    # --- training loop -----------------------------------------------------
    episode_rewards = []
    for episode in range(n_episodes):
        state = env.reset()        
        done, reward_tot = False, 0.0

        
        while not done:
            action, action_prob = agent.get_action(state)
            prev_state = state

            state, reward, done, info = env.step(action.detach().cpu().numpy())

            # dispatch sul corretto signature di store_outcome
            if agent_name == "REINFORCE":
                # vanilla REINFORCE: store_outcome(self, action_log_prob, reward)
                agent.store_outcome(action_prob, reward)

            elif agent_name == "REINFORCE_BAVG":
                # same signature della vanilla
                agent.store_outcome(action_prob, reward)

            elif agent_name == "REINFORCE_BVAL":
                # il tuo REINFORCE_baseline_value_net.py definisce
                # store_outcome(self, state, action_log_prob, reward)
                agent.store_outcome(prev_state, action_prob, reward)

            elif agent_name == "ActorCritic":
                # ActorCritic ha signature
                # store_outcome(self, prev_state, state, action_log_prob, reward, done)
                agent.store_outcome(prev_state, state, action_prob, reward, done)

            else:
                raise ValueError(f"Unknown agent {agent_name}")

            reward_tot += reward


        agent.update_policy()
        episode_rewards.append(reward_tot)

        if (episode + 1) % 100 == 0:
            print(f"Episode {episode + 1}: return = {reward_tot:.2f}")

    # --- save -------------------------------------------------------------


    mw = os.path.join(BASE_DIR, "models_weights")
    md = os.path.join(BASE_DIR, "models_data")
    if not os.path.isdir(mw):
        os.makedirs(mw)
    if not os.path.isdir(md):
        os.makedirs(md)

    filename_prefix = {
        "REINFORCE"      : "vanilla",
        "REINFORCE_BAVG" : "bavg",
        "REINFORCE_BVAL" : "bval"
    }[agent_name]

    torch.save(
        agent.policy.state_dict(),
        os.path.join(mw, f"{filename_prefix}_seed_{seed}_model.mdl")
    )


    episodes = np.arange(1, len(episode_rewards) + 1)
    rewards  = np.array(episode_rewards)

    data = np.column_stack((episodes, rewards))
    np.savetxt(
        os.path.join(
            md,
            f"{filename_prefix}_seed_{seed}_returns.csv"
        ),
        data,
        delimiter=",",
        header="episode,return",
        comments=""
    )
    
if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Train REINFORCE variants or ActorCritic on Hopper"
    )
    p.add_argument("--seed",     type=int, required=True,
                   help="Random seed for reproducibility")
    p.add_argument("--agent",    required=True,
                   choices=["REINFORCE", "REINFORCE_BAVG", "REINFORCE_BVAL", "ActorCritic"],
                   help="Which algorithm/variant to train")
    p.add_argument("--episodes", type=int, required=True,
                   help="Number of training episodes")
    p.add_argument("--device",   choices=["cpu","cuda"], default="cpu",
                   help="Compute device")
    args = p.parse_args()

    run_train(
        agent_name = args.agent,
        n_episodes = args.episodes,
        device     = args.device,
        seed       = args.seed
    )


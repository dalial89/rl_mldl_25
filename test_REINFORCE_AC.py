"""Test an RL agent on the OpenAI Gym Hopper environment"""
import os
BASE_DIR = os.path.dirname(os.path.realpath(__file__))

import importlib
import sys
import argparse

import torch
import gym
import random

from env.custom_hopper import *


def import_agent_module(agent_name: str):
    # Map flag -> module path
    pkg = {
        "REINFORCE"      : "REINFORCE.REINFORCE",
        "REINFORCE_BAVG" : "REINFORCE.REINFORCE_baseline_avg",
        "REINFORCE_BVAL" : "REINFORCE.REINFORCE_baseline_value_net",
        "ActorCritic"    : "ActorCritic.ActorCritic",
    }.get(agent_name)
    if pkg is None:
        raise ValueError(f"[ERROR] agent '{agent_name}' not recognised")
    try:
        module = importlib.import_module(f"agentsandpolicies.{pkg}")
    except ModuleNotFoundError as exc:
        print(f"[ERROR] module '{pkg}' not found!", file=sys.stderr)
        raise exc
    # Qui restituiamo davvero Policy e Agent
    if not all(hasattr(module, cls) for cls in ("Policy", "Agent")):
        raise AttributeError(f"[ERROR] {pkg} deve esporre Policy e Agent")
    return module.Policy, module.Agent


def run_test(
    agent_name: str,
    n_episodes: int,
    device: str,
    render:bool,
    seed: int
):
    
    # --- environment -------------------------------------------------------
    env = gym.make("CustomHopper-source-v0")
    # env = gym.make("CustomHopper-target-v0")

    print("Action space:", env.action_space)
    print("State space :", env.observation_space)
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

    obs_dim = env.observation_space.shape[-1]
    act_dim = env.action_space.shape[-1]

    policy = PolicyClass(obs_dim, act_dim).to(device)

    prefix = {
        "REINFORCE"      : "vanilla",
        "REINFORCE_BAVG" : "bavg",
        "REINFORCE_BVAL" : "bval",
        "ActorCritic"    : "actorcritic",
    }[agent_name]
    model_file = f"{prefix}_seed_{seed}_model.mdl"
    model_path = os.path.join(BASE_DIR, "models_weights", model_file)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"[ERROR] file of weights '{model_path}' not found!")

    policy.load_state_dict(torch.load(model_path, map_location=device))

    agent = AgentClass(policy, device=device)

    # --- evaluation loop ---------------------------------------------------
    returns = []
    for episode in range(n_episodes):
        state = env.reset()
        done, test_reward = False, 0.0

        while not done:
            with torch.no_grad():
                action, _ = agent.get_action(state, evaluation=True)

            state, reward, done, info = env.step(action.cpu().numpy())
            if render:
                env.render()
            test_reward += reward

        returns.append(test_reward)

        print(f"Episode: {episode} | Return: {test_reward:.2f}")

    md = os.path.join(BASE_DIR, "models_data")
    data = np.array(returns)
    np.savetxt(
        os.path.join(
            md,
            f"{agent_name}_seed_{seed}_test_returns.csv"
        ),
        data,
        delimiter=",",
        header="return",
        comments=""
    )



if __name__ == "__main__":

    p = argparse.ArgumentParser(description="Test REINFORCE/ActorCritic on Hopper")
    p.add_argument("--seed",     type=int, required=True,
                   help="Random seed for reproducibility")
    p.add_argument("--agent",    required=True,
                   choices=["REINFORCE","REINFORCE_BAVG","REINFORCE_BVAL","ActorCritic"],
                   help="type agent to test")
    p.add_argument("--episodes", type=int, default=10,
                   help="number of episodes of test")
    p.add_argument("--device",   choices=["cpu","cuda"], default="cpu",
                   help="Device")
    p.add_argument("--render",   action="store_true",
                   help="want to render")
    args = p.parse_args()

    run_test(
        agent_name = args.agent,
        n_episodes = args.episodes,
        device     = args.device,
        render     = args.render,
        seed       = args.seed
    )




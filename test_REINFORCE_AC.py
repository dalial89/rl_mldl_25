"""Test an RL agent on the OpenAI Gym Hopper environment"""
import importlib
import sys
from pathlib import Path

import torch
import gym

from env.custom_hopper import *


def import_agent_module(agent_name: str):
    try:
        module = importlib.import_module(f"agentsandpolicies.{agent_name}.{agent_name}")
    except ModuleNotFoundError as exc:
        print(f"[ERROR] agent '{agent_name}' not found!", file=sys.stderr)
        raise exc

    if not all(hasattr(module, cls) for cls in ("Policy", "Agent")):
        raise AttributeError(
            f"[ERROR] {agent_name} should expose 'Policy' and 'Agent' classes"
        )
    return module.Policy, module.Agent


def run_test(
    agent_name: str,
    n_episodes: int,
    device: str,
    render:bool,
    baseline:bool,
    eps:float
):

    # --- environment -------------------------------------------------------
    env = gym.make("CustomHopper-source-v0")
    # env = gym.make("CustomHopper-target-v0")

    print("Action space:", env.action_space)
    print("State space :", env.observation_space)
    print("Dynamics parameters:", env.get_parameters())

    # --- import agent ------------------------------------------------------
    PolicyClass, AgentClass = import_agent_module(agent_name)

    obs_dim = env.observation_space.shape[-1]
    act_dim = env.action_space.shape[-1]

    policy = PolicyClass(obs_dim, act_dim).to(device)

    model_path = Path(
        f"models_weights/{agent_name}_baseline_{baseline}_eps_{eps}_model.mdl"
    )

    if not model_path.exists():
        raise FileNotFoundError(
            f"[ERROR] weights file '{model_path}' not found"
        )

    policy.load_state_dict(torch.load(model_path, map_location=device))

    agent = AgentClass(
        policy,
        device=device,
        baseline=baseline,
        eps=eps
    )

    # --- evaluation loop ---------------------------------------------------
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

        print(f"Episode: {episode} | Return: {test_reward:.2f}")

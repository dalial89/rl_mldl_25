"""Test an RL agent on the OpenAI Gym Hopper environment"""
import importlib
import sys
from pathlib import Path

import torch
import gym
import random

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
    eps:float,
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

if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(
        description="Test REINFORCE/ActorCritic on Hopper"
    )
    p.add_argument("--seed", type=int, default=42,
               help="Random seed for reproducibility")
    p.add_argument("--agent",    required=True,
                   choices=["REINFORCE","ActorCritic"],
                   help="Which agent to test")
    p.add_argument("--episodes", type=int,    default=10,
                   help="Number of evaluation episodes")
    p.add_argument("--device",   choices=["cpu","cuda"],
                   default="cpu",
                   help="Device for evaluation")
    p.add_argument("--baseline", action="store_true",
                   help="Whether the trained model used a mean–std baseline")
    p.add_argument("--eps",      type=float,
                   default=1e-8,
                   help="Epsilon that was used for baseline normalization")
    p.add_argument("--render",   action="store_true",
                   help="Render the environment during testing")
    args = p.parse_args()

    run_test(
        agent_name = args.agent,
        n_episodes = args.episodes,
        device     = args.device,
        render     = args.render,
        baseline   = args.baseline,
        eps        = args.eps
    )


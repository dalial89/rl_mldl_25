import numpy as np
import gym, random, torch, argparse
from pathlib import Path
import pandas as pd

from env.custom_hopper import *                       # noqa: F401,F403
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize


# 1) Utilities

def set_seeds(seed: int):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def make_eval_env(env_id: str, seed: int):
    env_raw = gym.make(env_id)
    env_raw.seed(seed)
    env = Monitor(env_raw, filename=None)
    return DummyVecEnv([lambda: env])

# 2) Test a single model–env pair

def test_sb3_model(model_path, env_id, run_tag, episodes, render, device, seed):
    vec_env = make_eval_env(env_id, seed)

    
    logs_dir  = Path(__file__).resolve().parent / f"logs_{run_tag}_seed{seed}"
    norm_path = logs_dir / "vecnormalize.pkl"
    if norm_path.exists():
        vec_env = VecNormalize.load(str(norm_path), vec_env)
        vec_env.training = False
        vec_env.norm_reward = False
    else:
        print(f"Warning: no VecNormalize at {norm_path}, evaluating un-normalized.")

    model = PPO.load(str(model_path), env=vec_env, device=device)

    returns = []
    for ep in range(1, episodes + 1):
        obs, done, total_reward = vec_env.reset(), False, 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _ = vec_env.step(action)
            total_reward += float(reward)
            if render:
                vec_env.render()
        returns.append(total_reward)
        print(f"Episode {ep}: Return = {total_reward:.2f}")

    vec_env.close()
    return np.mean(returns), np.std(returns)


# 3) Main entry-point (callable from launcher)

def run_tests(*, seed: int, use_udr: bool, render: bool = False,
              device: str = "cpu", episodes: int = 100) -> None:
    """Esegue i tre test-case e salva un CSV con i risultati."""
    set_seeds(seed)

    PROJ_ROOT  = Path(__file__).resolve().parent.parents[1]
    MODELS_DIR = PROJ_ROOT / "models_weights"
    OUT_DIR    = PROJ_ROOT / "training_data"
    OUT_DIR.mkdir(exist_ok=True)

    test_cases = [
        ("source→source", "source_udr" if use_udr else "source_base", "CustomHopper-source-v0"),
        ("source→target", "source_udr" if use_udr else "source_base", "CustomHopper-target-v0"),
        ("target→target", "target_base",                              "CustomHopper-target-v0"),
    ]

    print(f"\nEvaluating {episodes} episodes per case (render={render}, seed={seed})\n")
    rows = []

    for label, run_tag, env_id in test_cases:
        model_path = MODELS_DIR / f"{run_tag}_seed{seed}_final.zip"
        if not model_path.exists():
            print(f"Model not found: {model_path}")
            continue

        mean_ret, std_ret = test_sb3_model(model_path, env_id, run_tag,
                                           episodes, render, device, seed)
        print(f"{label:<14} | Env: {env_id:<24} | "
              f"Mean: {mean_ret:7.2f} ± {std_ret:7.2f}")
        rows.append(dict(setup=label, env=env_id, seed=seed,
                         udr=use_udr, mean=mean_ret, std=std_ret,
                         episodes=episodes))

    #save csv
    if rows:
        out_file = OUT_DIR / f"test_seed{seed}_{'udr' if use_udr else 'base'}.csv"
        pd.DataFrame(rows).to_csv(out_file, index=False)
        print(f"\nSaved test results to {out_file}\n")




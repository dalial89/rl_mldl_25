import argparse
import subprocess
import sys
import random
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Run tuning, training, or evaluation"
    )
    parser.add_argument(
        "--ppo_tuning",
        action="store_true",
        help="Run the standard PPO hyperparameter sweep"
    )
    parser.add_argument(
        "--udr_tuning",
        action="store_true",
        help="Run the UDR mass-randomization sweep"
    )
    parser.add_argument(
        "--run_training",
        action="store_true",
        help="Run final PPO training with best UDR + hyperparameters"
    )
    parser.add_argument(
        "--run_testing",
        action="store_true",
        help="Run evaluation/testing of trained vanilla and UDR models"
    )
    parser.add_argument(
        "--use-udr",
        action="store_true",
        help="Enable UDR when training or testing"
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Render the envs while running evaluation"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed forwarded to scripts"
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Compute device forwarded to sub-scripts"
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=50,
        help="Number of episodes for evaluation scripts"
    )
    # SimOpt flags
    parser.add_argument(
        "--simopt_train",
        action="store_true",
        help="Run adaptive SimOpt training (Bayesian)"
    )
    parser.add_argument(
        "--simopt_test",
        action="store_true",
        help="Run evaluation/testing of SimOpt models (Bayesian)"
    )
    # PSO-based SimOpt (train + test)
    parser.add_argument(
        "--simopt_pso",
        action="store_true",
        help="Run PSO-based SimOpt training and evaluation"
    )
    args = parser.parse_args()

    print("\n===== RL-MLDL-25 Pipeline Launcher =====\n")

    # Require at least one action
    if not any([
        args.ppo_tuning, args.udr_tuning, args.run_training,
        args.run_testing, args.simopt_train, args.simopt_test,
        args.simopt_pso
    ]):
        parser.error(
            "Please specify an action: --ppo_tuning, --udr_tuning,"
            " --run_training, --run_testing, --simopt_train,"
            " --simopt_test or --simopt_pso"
        )

    # PPO hyperparameter sweep
    if args.ppo_tuning:
        print(">>> Starting PPO hyperparameter sweep...")
        from tuning.PPO_tuning import main as ppo_main
        ppo_main()
        print(">>> PPO hyperparameter sweep completed.\n")

    # UDR mass-randomization sweep
    if args.udr_tuning:
        print(">>> Starting UDR mass-randomization sweep...")
        from tuning.PPO_UDR_tuning import main as udr_main
        udr_main()
        print(">>> UDR mass-randomization sweep completed.\n")

    # Final PPO training (vanilla or UDR)
    if args.run_training:
        print(">>> Starting final PPO training with best UDR + configs...")
        from agentsandpolicies.PPOandUDR.run_trainingPPOandUDR import main as train_main
        train_main(seed=args.seed, use_udr=args.use_udr)
        print(">>> Final PPO training completed.\n")

    # Evaluation/testing of trained vanilla/UDR models
    if args.run_testing:
        print(">>> Starting evaluation/testing of trained models...")
        from agentsandpolicies.PPOandUDR.run_testingPPOandUDR import run_tests as test_main
        test_main(seed=args.seed, use_udr=args.use_udr, render=args.render)
        print(">>> Evaluation/testing completed.\n")

    # Bayesian SimOpt training
    if args.simopt_train:
        print(">>> Starting adaptive SimOpt training (Bayesian)...")
        from agentsandpolicies.PPOandSimOpt.trainSimOpt import run_simopt
        run_simopt(seed=args.seed, device=args.device)
        print(">>> Bayesian SimOpt training completed.\n")

    # Bayesian SimOpt evaluation
    if args.simopt_test:
        print(">>> Starting evaluation/testing of Bayesian SimOpt models...")
        script_path = Path(__file__).resolve().parent \
            / "agentsandpolicies" / "PPOandSimOpt" / "testSimOpt.py"
        cmd = [
            sys.executable, str(script_path),
            "--seed", str(args.seed),
            "--device", args.device,
            "--episodes", str(args.episodes)
        ]
        if args.render:
            cmd.append("--render")
        if args.use_udr:
            cmd.append("--udr")
        print("[subprocess]", " ".join(cmd))
        subprocess.call(cmd)
        print(">>> Bayesian SimOpt evaluation completed.\n")

    # PSO-based SimOpt (train + test)
    if args.simopt_pso:
        print(">>> Starting PSO-based SimOpt training...")
        from agentsandpolicies.PPOandSimOpt.trainSimOptPSO import main as pso_main
        pso_main(seed=args.seed, device=args.device)
        print(">>> PSO-based SimOpt training completed.\n")

        print(">>> Starting PSO-based SimOpt evaluation...")
        script_path = Path(__file__).resolve().parent \
            / "agentsandpolicies" / "PPOandSimOpt" / "testSimOpt.py"
        cmd = [
            sys.executable, str(script_path),
            "--seed", str(args.seed),
            "--device", args.device,
            "--episodes", str(args.episodes)
        ]
        if args.render:
            cmd.append("--render")
        if args.use_udr:
            cmd.append("--udr")
        print("[subprocess]", " ".join(cmd))
        subprocess.call(cmd)
        print(">>> PSO-based SimOpt evaluation completed.\n")

    print("===== Pipeline finished =====")


if __name__ == "__main__":
    main()

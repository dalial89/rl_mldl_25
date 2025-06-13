import argparse
import subprocess
import sys
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
        help="Run evaluation/testing of trained models (vanilla & UDR)"
    )
    parser.add_argument(
        "--use-udr",
        action="store_true",
        help="Enable UDR when training (forwarded to the training script)"
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
        help="Number of episodes for SimOpt test"
    )
    # SimOpt flags
    parser.add_argument(
        "--simopt_train",
        action="store_true",
        help="Run adaptive SimOpt training"
    )
    parser.add_argument(
        "--simopt_test",
        action="store_true",
        help="Run evaluation/testing of SimOpt models"
    )
    args = parser.parse_args()

    print("\n===== RL-MLDL-25 Pipeline Launcher =====\n")

    if not any([
        args.ppo_tuning, args.udr_tuning, args.run_training,
        args.run_testing, args.simopt_train, args.simopt_test
    ]):
        parser.error(
            "Please specify at least one action: --ppo_tuning, --udr_tuning,"
            " --run_training, --run_testing, --simopt_train or --simopt_test"
        )

    if args.ppo_tuning:
        print(">>> Starting PPO hyperparameter sweep...")
        from tuning.PPO_tuning import main as ppo_main
        ppo_main()
        print(">>> PPO hyperparameter sweep completed.\n")

    if args.udr_tuning:
        print(">>> Starting UDR mass-randomization sweep...")
        from tuning.PPO_UDR_tuning import main as udr_main
        udr_main()
        print(">>> UDR mass-randomization sweep completed.\n")

    if args.run_training:
        print(">>> Starting final training with best UDR + PPO configs...")
        from agentsandpolicies.PPOandUDR.run_trainingPPOandUDR import main as train_main
        train_main(seed=args.seed, use_udr=args.use_udr)
        print(">>> Final PPO training completed.\n")

    if args.run_testing:
        print(">>> Starting evaluation/testing of trained models...")
        from agentsandpolicies.PPOandUDR.run_testingPPOandUDR import run_tests as test_main
        test_main(seed=args.seed, use_udr=args.use_udr, render=args.render)
        print(">>> Evaluation/testing completed.\n")

    if args.simopt_train:
        print(">>> Starting adaptive SimOpt training...")
        from agentsandpolicies.PPOandSimOpt.trainSimOpt import run_simopt
        run_simopt(seed=args.seed, device=args.device)
        print(">>> SimOpt training completed.\n")

    if args.simopt_test:
        print(">>> Starting evaluation/testing of SimOpt models...")
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
        print(">>> SimOpt evaluation completed.\n")

    print("===== Pipeline finished =====")


if __name__ == "__main__":
    main()


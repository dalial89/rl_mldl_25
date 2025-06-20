import argparse
import subprocess
import sys
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(
        description="RL-MLDL-25 Pipeline Launcher"
    )
    # High-level actions
    parser.add_argument("--ppo_tuning",   action="store_true",
                        help="Run the standard PPO hyperparameter sweep")
    parser.add_argument("--udr_tuning",   action="store_true",
                        help="Run the UDR mass-randomization sweep")
    parser.add_argument("--run_training", action="store_true",
                        help="Run training for the selected agent")
    parser.add_argument("--run_testing",  action="store_true",
                        help="Run evaluation/testing of trained models")
    parser.add_argument("--simopt_train", action="store_true",
                        help="Run adaptive SimOpt training (Bayesian)")
    parser.add_argument("--simopt_test",  action="store_true",
                        help="Run evaluation/testing of Bayesian SimOpt models")
    parser.add_argument("--simopt_pso",   action="store_true",
                        help="Run PSO-based SimOpt training and evaluation")

    # Common options
    parser.add_argument("--agent",    choices=["REINFORCE","ActorCritic","PPO"],
                        default="REINFORCE",
                        help="Which agent to train/test")
    parser.add_argument("--use-udr",  action="store_true",
                        help="Enable UDR when training or testing (PPO)")
    parser.add_argument("--baseline", action="store_true",
                        help="Enable mean-std baseline (for REINFORCE/ActorCritic)")
    parser.add_argument("--eps",      type=float, default=1e-8,
                        help="Epsilon for baseline normalization")
    parser.add_argument("--seed",     type=int, default=42,
                        help="Random seed forwarded to sub-scripts")
    parser.add_argument("--device",   choices=["cpu","cuda"], default="cpu",
                        help="Compute device forwarded to sub-scripts")
    parser.add_argument("--episodes", type=int, default=100_000,
                        help="Number of training or evaluation episodes")
    parser.add_argument("--render",   action="store_true",
                        help="Render the envs during evaluation")
    return parser.parse_args()


def main():
    args = parse_args()
    print("\n===== RL-MLDL-25 Pipeline Launcher =====\n")

    # Require at least one action
    if not any([
        args.ppo_tuning, args.udr_tuning,
        args.run_training, args.run_testing,
        args.simopt_train, args.simopt_test,
        args.simopt_pso
    ]):
        sys.exit("Error: specify at least one of --ppo_tuning, --udr_tuning, "
                 "--run_training, --run_testing, --simopt_train, --simopt_test, "
                 "or --simopt_pso")

    # 1) PPO hyperparameter sweep
    if args.ppo_tuning:
        print(">>> Starting PPO hyperparameter sweep...")
        from tuning.PPO_tuning import main as ppo_main
        ppo_main()
        print(">>> PPO hyperparameter sweep completed.\n")

    # 2) UDR mass-randomization sweep
    if args.udr_tuning:
        print(">>> Starting UDR mass-randomization sweep...")
        from tuning.PPO_UDR_tuning import main as udr_main
        udr_main()
        print(">>> UDR mass-randomization sweep completed.\n")

    # 3) Training 
    if args.run_training:
        print(f">>> Starting training for {args.agent} with baseline = {args.baseline} (eps={args.eps}) (episodes={args.episodes})...")
        if args.agent in ("REINFORCE", "ActorCritic"):
            script_tr = (Path(__file__).resolve().parent / "train_REINFORCE_AC.py")
            cmd_tr = [
                sys.executable, str(script_tr),
                "--agent",    args.agent,
                "--episodes", str(args.episodes),
                "--device",   args.device
            ]
            if args.baseline:
                cmd_tr.append("--baseline")
                cmd_tr += ["--eps", str(args.eps)]

            print("[subprocess]", " ".join(cmd_tr))
            subprocess.call(cmd_tr)

        elif args.agent == "PPO":
            print(">>> Starting final PPO training...")
            from agentsandpolicies.PPOandUDR.run_trainingPPOandUDR import main as train_main
            train_main(seed=args.seed, use_udr=args.use_udr)
        else:
            sys.exit(f"Unknown agent '{args.agent}'")

    # 4) Manual testing if requested
    if args.run_testing:
        print(f">>> Starting evaluation for {args.agent}...")
        if args.agent in ("REINFORCE", "ActorCritic"):
            script = (Path(__file__).resolve().parent
                      / "test_REINFORCE_AC.py")
            cmd = [
                sys.executable, str(script),
                "--agent",    args.agent,
                "--episodes", str(args.episodes),
                "--device",   args.device
            ]
            if args.baseline:
                cmd.append("--baseline")
                cmd += ["--eps", str(args.eps)]
            if args.render:
                cmd.append("--render")
            print("[subprocess]", " ".join(cmd))
            subprocess.call(cmd)

        elif args.agent == "PPO":
            from agentsandpolicies.PPOandUDR.run_testingPPOandUDR import run_tests as test_main
            test_main(seed=args.seed, use_udr=args.use_udr, render=args.render)
        else:
            sys.exit(f"Unknown agent '{args.agent}'")
        print(f">>> {args.agent} evaluation completed.\n")

    # 5) Bayesian SimOpt training
    if args.simopt_train:
        print(">>> Starting adaptive SimOpt training (Bayesian)...")
        from agentsandpolicies.PPOandSimOpt.trainSimOpt import run_simopt
        run_simopt(seed=args.seed, device=args.device)
        print(">>> Bayesian SimOpt training completed.\n")

    # 6) Bayesian SimOpt evaluation
    if args.simopt_test:
        print(">>> Starting evaluation/testing of Bayesian SimOpt models...")
        script = (Path(__file__).resolve().parent
                  / "agentsandpolicies" / "PPOandSimOpt" / "testSimOpt.py")
        cmd = [
            sys.executable, str(script),
            "--seed",     str(args.seed),
            "--device",   args.device,
            "--episodes", str(args.episodes)
        ]
        if args.render:
            cmd.append("--render")
        if args.use_udr:
            cmd.append("--udr")
        print("[subprocess]", " ".join(cmd))
        subprocess.call(cmd)
        print(">>> Bayesian SimOpt evaluation completed.\n")

    # 7) PSO-based SimOpt (train + test)
    if args.simopt_pso:
        print(">>> Starting PSO-based SimOpt training...")
        from agentsandpolicies.PPOandSimOpt.trainSimOptPSO import main as pso_main
        pso_main(seed=args.seed, device=args.device)
        print(">>> PSO-based SimOpt training completed.\n")

        print(">>> Starting PSO-based SimOpt evaluation...")
        script = (Path(__file__).resolve().parent
                  / "agentsandpolicies" / "PPOandSimOpt" / "testSimOpt.py")
        cmd = [
            sys.executable, str(script),
            "--seed",     str(args.seed),
            "--device",   args.device,
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


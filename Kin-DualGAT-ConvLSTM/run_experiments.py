import argparse
import subprocess


def run_ablation():
    print("Running factorial ablations (Table 11)...")
    # subprocess.call to train.py with config overrides
    pass


def run_robustness():
    print("Running robustness tests with Gaussian noise and coordinate loss (Table 10)...")
    pass


def run_bootstrap():
    print("Calculating bootstrap confidence intervals for Table 8...")
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["ablation", "robustness", "bootstrap"], required=True)
    args = parser.parse_args()

    if args.mode == "ablation":
        run_ablation()
    elif args.mode == "robustness":
        run_robustness()
    elif args.mode == "bootstrap":
        run_bootstrap()
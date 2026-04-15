"""
main_run.py
-----------
One-click script: data merge -> all models x all countries x all lags.

IMPORTANT: Before running this script, you must first collect raw data:
    python scripts/collect_data.py
This step requires a FRED API key and manual downloads:
    - Shanghai_Composite.csv (Investing.com) -> data/raw/
    - uk_m1_emp.csv (BOE/ONS) -> data/raw/
    - de_price_compet.csv (Deutsche Bundesbank) -> data/raw/

Usage:
    python scripts/main_run.py                  # full run (merge + all)
    python scripts/main_run.py --skip-merge     # skip data merge step
    python scripts/main_run.py --lags 0         # only lag=0
    python scripts/main_run.py --countries united_kingdom --models mlp xgboost
"""

import argparse
import subprocess
import sys
import os
import time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS_DIR = os.path.join(PROJECT_ROOT, "scripts")

COUNTRIES = ["china", "united_states", "united_kingdom", "germany"]
MODELS = ["chow_lin", "elastic_net", "xgboost", "mlp"]  # fastest first
LAG_VALUES = [0, 1, 2]


def run_cmd(cmd, description):
    """Run a command and print status."""
    print(f"\n{'='*60}")
    print(f"  {description}")
    print(f"{'='*60}")
    start = time.time()
    result = subprocess.run(
        [sys.executable] + cmd,
        cwd=PROJECT_ROOT,
    )
    elapsed = time.time() - start
    status = "OK" if result.returncode == 0 else "FAILED"
    print(f"  [{status}] {description} ({elapsed/60:.1f} min)")
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(description="Full pipeline: merge + run all")
    parser.add_argument("--skip-merge", action="store_true",
                        help="Skip data merge step")
    parser.add_argument("--lags", type=int, nargs="+", default=LAG_VALUES,
                        help="Lag values to run (default: 0 1 2)")
    parser.add_argument("--countries", type=str, nargs="+", default=COUNTRIES,
                        help="Countries to run")
    parser.add_argument("--models", type=str, nargs="+", default=MODELS,
                        help="Models to run")
    args = parser.parse_args()

    total_start = time.time()
    results = []

    # Step 1: Data merge
    if not args.skip_merge:
        ok = run_cmd(
            [os.path.join(SCRIPTS_DIR, "merge_data.py")],
            "Data merge + summary stats + ADF"
        )
        if not ok:
            print("Data merge failed. Aborting.")
            sys.exit(1)

    # Step 2: Run all combinations
    for lag in args.lags:
        for country in args.countries:
            for model in args.models:
                desc = f"{country} | {model} | lag={lag}"
                ok = run_cmd(
                    [os.path.join(SCRIPTS_DIR, "run_country.py"),
                     "--country", country,
                     "--model", model,
                     "--lags", str(lag)],
                    desc
                )
                results.append((desc, "OK" if ok else "FAILED"))

    # Step 3: Compile results
    print("\n" + "=" * 60)
    print("  Compiling results...")
    print("=" * 60)
    run_cmd(
        [os.path.join(SCRIPTS_DIR, "compile_results.py")],
        "Compile summary + DM tests"
    )
    
    # Summary
    total_elapsed = time.time() - total_start
    print(f"\n{'='*60}")
    print(f"  ALL DONE  ({total_elapsed/3600:.1f} hours)")
    print(f"{'='*60}")
    for desc, status in results:
        marker = "+" if status == "OK" else "X"
        print(f"  [{marker}] {desc}")

    failed = sum(1 for _, s in results if s == "FAILED")
    if failed > 0:
        print(f"\n  {failed} task(s) failed.")


if __name__ == "__main__":
    main()
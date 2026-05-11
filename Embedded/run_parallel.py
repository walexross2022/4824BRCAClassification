"""
Embedded/run_parallel.py

Spawns 10 worker processes for LASSO and/or Random Forest embedded methods
across specified datasets under 10 different random seeds.

Each (method, run_id, seed) job writes a per-run CSV to:
    Embedded/results/runs/<method>/run_<NN>.csv

After execution, run `aggregate_results.py` to merge runs into mean/std
summaries.

Usage:
    python run_parallel.py --methods lasso           # LASSO only
    python run_parallel.py --methods rf              # RF only
    python run_parallel.py --methods lasso rf        # both
    python run_parallel.py --n-runs 5                # custom rep count
    python run_parallel.py --cancers COAD PRAD       # skip BRCA
"""

import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import pandas as pd

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(THIS_DIR)
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, THIS_DIR)
sys.path.insert(0, os.path.join(BASE_DIR, "Filter"))

RUNS_DIR = os.path.join(THIS_DIR, "results", "runs")
CANCERS = ["BRCA", "COAD", "PRAD"]


def _set_single_threaded():
    """Limit BLAS / OpenMP threads inside workers to avoid CPU oversubscription."""
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS",
                "OPENBLAS_NUM_THREADS", "BLIS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS"):
        os.environ[var] = "1"


def _worker(method, run_id, seed, cancer):
    """Run one method on one cancer under one seed. Writes one CSV."""
    _set_single_threaded()

    import warnings
    warnings.filterwarnings("ignore")

    out_dir = os.path.join(RUNS_DIR, method)
    os.makedirs(out_dir, exist_ok=True)

    start = time.time()

    if method == "lasso":
        from run_lasso import run_lasso_workflow
        res = run_lasso_workflow(
            cancer_type=cancer,
            seed=seed,
            save_plots=False,
            use_cache=True,
            n_jobs=1,
        )
        df = pd.DataFrame([res])
        df["Cancer"] = cancer
        df["Method"] = "LASSO"
        df["RunID"] = run_id

    elif method == "rf":
        from run_rf import run_rf_workflow
        res = run_rf_workflow(
            cancer_type=cancer,
            seed=seed,
            save_plots=False,
            use_cache=True,
            n_jobs=1,
        )
        df = pd.DataFrame([res])
        df["Cancer"] = cancer
        df["Method"] = "RandomForest"
        df["RunID"] = run_id
    else:
        raise ValueError(f"Unknown method: {method}")

    out_path = os.path.join(out_dir, f"run_{run_id:02d}_{cancer}.csv")
    df.to_csv(out_path, index=False)

    elapsed = time.time() - start
    return method, run_id, cancer, elapsed, out_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--methods", nargs="+", default=["lasso", "rf"],
                        choices=["lasso", "rf"])
    parser.add_argument("--cancers", nargs="+", default=CANCERS,
                        choices=CANCERS)
    parser.add_argument("--n-runs", type=int, default=10)
    parser.add_argument("--max-workers", type=int, default=10)
    parser.add_argument("--base-seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(RUNS_DIR, exist_ok=True)

    rng = __import__("numpy").random.default_rng(args.base_seed)
    seeds = [int(s) for s in rng.integers(0, 2**31 - 1, size=args.n_runs)]

    for method in args.methods:
        print(f"\n{'=' * 60}")
        print(f"  Method: {method}  |  cancers: {args.cancers}  |  "
              f"runs: {args.n_runs}  |  workers: {args.max_workers}")
        print(f"{'=' * 60}")

        method_start = time.time()

        # Submit one job per (run_id, cancer)
        jobs = []
        for run_id, seed in enumerate(seeds):
            for cancer in args.cancers:
                jobs.append((method, run_id, seed, cancer))

        with ProcessPoolExecutor(max_workers=args.max_workers) as ex:
            futures = {
                ex.submit(_worker, *job): job for job in jobs
            }
            for fut in as_completed(futures):
                job = futures[fut]
                try:
                    m, rid, can, elapsed, out_path = fut.result()
                    print(f"  [{m}] run_{rid:02d}_{can} done in "
                          f"{elapsed:.1f}s")
                except Exception as e:
                    print(f"  [{method}] run_{job[1]:02d}_{job[3]} "
                          f"FAILED: {e}")

        method_elapsed = time.time() - method_start
        print(f"  {method} total wall time: {method_elapsed:.1f}s")


if __name__ == "__main__":
    main()

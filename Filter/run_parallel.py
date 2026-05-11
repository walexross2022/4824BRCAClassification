"""
Filter/run_parallel.py

Spawns 10 worker processes per filter method (ANOVA, Mutual Information,
Variance Threshold) and runs each method on all three datasets (BRCA, COAD,
PRAD) under 10 different random seeds.

Each (method, seed) job writes a per-run CSV to:
    Filter/results/runs/<method>/run_<NN>.csv

After execution, run `aggregate_results.py` to merge runs into mean/std
summaries.

Usage:
    python run_parallel.py                  # all methods, 10 runs each
    python run_parallel.py --methods anova  # one method
    python run_parallel.py --n-runs 5       # custom rep count
    python run_parallel.py --max-workers 8  # cap worker count
"""

import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import pandas as pd

# Ensure project root is importable when running this file directly
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(THIS_DIR)
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, THIS_DIR)


RUNS_DIR = os.path.join(THIS_DIR, "results", "runs")
CANCERS = ["BRCA", "COAD", "PRAD"]
METHODS = ["anova", "mutual_info", "variance"]


def _set_single_threaded():
    """Limit BLAS / OpenMP threads inside workers to avoid CPU oversubscription."""
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS",
                "OPENBLAS_NUM_THREADS", "BLIS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS"):
        os.environ[var] = "1"


def _worker(method, run_id, seed):
    """Run one method across all 3 cancers under one seed. Writes one CSV."""
    _set_single_threaded()

    import warnings
    warnings.filterwarnings("ignore")

    out_dir = os.path.join(RUNS_DIR, method)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"run_{run_id:02d}.csv")

    start = time.time()

    rows = []
    if method == "anova":
        from anova_filter import run_anova_filter
        for cancer in CANCERS:
            df, _ = run_anova_filter(
                cancer_type=cancer,
                seed=seed,
                save_plots=False,
                use_cache=True,
                n_jobs=1,
            )
            df["RunID"] = run_id
            rows.append(df)

    elif method == "mutual_info":
        from mutual_info_filter import run_mutual_info_filter
        for cancer in CANCERS:
            df, _ = run_mutual_info_filter(
                cancer_type=cancer,
                seed=seed,
                save_plots=False,
                use_cache=True,
                n_jobs=1,
            )
            df["RunID"] = run_id
            rows.append(df)

    elif method == "variance":
        from variance_filter import run_variance_filter
        for cancer in CANCERS:
            df = run_variance_filter(
                cancer_type=cancer,
                seed=seed,
                save_plots=False,
                use_cache=True,
                n_jobs=1,
            )
            df["RunID"] = run_id
            rows.append(df)
    else:
        raise ValueError(f"Unknown method: {method}")

    combined = pd.concat(rows, ignore_index=True)
    combined.to_csv(out_path, index=False)

    elapsed = time.time() - start
    return method, run_id, elapsed, out_path


def run_method(method, n_runs, max_workers, base_seed=42):
    print(f"\n{'=' * 60}")
    print(f"  Method: {method}  |  runs: {n_runs}  |  workers: {max_workers}")
    print(f"{'=' * 60}")

    method_start = time.time()
    seeds = [base_seed + i for i in range(n_runs)]

    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futures = {
            ex.submit(_worker, method, run_id, seed): run_id
            for run_id, seed in enumerate(seeds)
        }
        for fut in as_completed(futures):
            run_id = futures[fut]
            try:
                m, rid, elapsed, out_path = fut.result()
                print(f"  [{m}] run_{rid:02d} done in {elapsed:.2f}s -> {out_path}")
            except Exception as e:
                print(f"  [{method}] run_{run_id:02d} FAILED: {e}")

    method_elapsed = time.time() - method_start
    print(f"  {method} total wall time: {method_elapsed:.2f}s")
    return method_elapsed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--methods", nargs="+", default=METHODS, choices=METHODS)
    parser.add_argument("--n-runs", type=int, default=10)
    parser.add_argument("--max-workers", type=int, default=10)
    parser.add_argument("--base-seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(RUNS_DIR, exist_ok=True)

    overall_start = time.time()
    timings = {}
    for method in args.methods:
        timings[method] = run_method(
            method=method,
            n_runs=args.n_runs,
            max_workers=args.max_workers,
            base_seed=args.base_seed,
        )

    total = time.time() - overall_start
    print("\n" + "=" * 60)
    print("  Summary")
    print("=" * 60)
    for m, t in timings.items():
        print(f"  {m:>14}: {t:>8.2f}s")
    print(f"  {'TOTAL':>14}: {total:>8.2f}s")


if __name__ == "__main__":
    main()

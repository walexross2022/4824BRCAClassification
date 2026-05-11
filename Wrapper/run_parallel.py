"""
Wrapper/run_parallel.py

Spawns 10 worker processes for the SelectKBest wrapper method and runs it
on all three datasets (BRCA, COAD, PRAD) under 10 different random seeds.

Each (run_id, seed) job writes a per-run CSV to:
    Wrapper/results/runs/selectkbest/run_<NN>.csv

After execution, run `aggregate_results.py` to merge runs into mean/std
summaries.

Usage:
    python run_parallel.py                  # 10 runs, all 3 datasets
    python run_parallel.py --n-runs 5       # custom rep count
    python run_parallel.py --max-workers 8  # cap worker count
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
METHOD = "selectkbest"


def _set_single_threaded():
    """Limit BLAS / OpenMP threads inside workers to avoid CPU oversubscription."""
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS",
                "OPENBLAS_NUM_THREADS", "BLIS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS"):
        os.environ[var] = "1"


def _worker(run_id, seed, fractions):
    """Run SelectKBest wrapper across all 3 cancers under one seed."""
    _set_single_threaded()

    import warnings
    warnings.filterwarnings("ignore")

    out_dir = os.path.join(RUNS_DIR, METHOD)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"run_{run_id:02d}.csv")

    start = time.time()

    from selectkbest_wrapper import run_selectkbest_wrapper

    rows = []
    for cancer in CANCERS:
        df = run_selectkbest_wrapper(
            cancer_type=cancer,
            seed=seed,
            fractions=fractions,
            save_plots=False,
            use_cache=True,
            n_jobs=1,
        )
        df["RunID"] = run_id
        rows.append(df)

    combined = pd.concat(rows, ignore_index=True)
    combined.to_csv(out_path, index=False)

    elapsed = time.time() - start
    return run_id, elapsed, out_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-runs", type=int, default=10)
    parser.add_argument("--max-workers", type=int, default=10)
    parser.add_argument("--base-seed", type=int, default=42)
    parser.add_argument("--fractions", type=float, nargs="+", default=[0.50])
    args = parser.parse_args()

    os.makedirs(RUNS_DIR, exist_ok=True)

    print(f"\n{'=' * 60}")
    print(f"  Method: {METHOD}  |  runs: {args.n_runs}  |  "
          f"workers: {args.max_workers}  |  fractions: {args.fractions}")
    print(f"{'=' * 60}")

    overall_start = time.time()
    seeds = [args.base_seed + i for i in range(args.n_runs)]

    with ProcessPoolExecutor(max_workers=args.max_workers) as ex:
        futures = {
            ex.submit(_worker, run_id, seed, args.fractions): run_id
            for run_id, seed in enumerate(seeds)
        }
        for fut in as_completed(futures):
            run_id = futures[fut]
            try:
                rid, elapsed, out_path = fut.result()
                print(f"  [{METHOD}] run_{rid:02d} done in {elapsed:.2f}s "
                      f"-> {out_path}")
            except Exception as e:
                print(f"  [{METHOD}] run_{run_id:02d} FAILED: {e}")

    total = time.time() - overall_start
    print("\n" + "=" * 60)
    print(f"  TOTAL wall time: {total:.2f}s")
    print("=" * 60)


if __name__ == "__main__":
    main()

"""
Wrapper/aggregate_results.py

Aggregate per-run CSVs produced by `run_parallel.py` into mean / std / min /
max summaries.

Reads:
    Wrapper/results/runs/<method>/run_<NN>.csv

Writes:
    Wrapper/results/aggregated/<method>_aggregated.csv
    Wrapper/results/aggregated/aggregated_summary.txt
"""

import glob
import os
import sys

import pandas as pd

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
RUNS_DIR = os.path.join(THIS_DIR, "results", "runs")
AGG_DIR = os.path.join(THIS_DIR, "results", "aggregated")

METRIC_COLS = ["Accuracy", "Macro F1", "Weighted F1", "ROC-AUC", "Runtime (s)"]


def _load_method_runs(method):
    paths = sorted(glob.glob(os.path.join(RUNS_DIR, method, "run_*.csv")))
    if not paths:
        return None
    frames = [pd.read_csv(p) for p in paths]
    return pd.concat(frames, ignore_index=True)


def _aggregate(df, group_cols):
    agg_funcs = ["mean", "std", "min", "max"]
    summary = df.groupby(group_cols)[METRIC_COLS].agg(agg_funcs)
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    n_runs = df.groupby(group_cols)["RunID"].nunique().rename("n_runs")
    summary = summary.join(n_runs)
    return summary.reset_index()


def aggregate_method(method):
    df = _load_method_runs(method)
    if df is None:
        print(f"[{method}] no run files found, skipping")
        return None

    group_cols = ["Cancer", "Method"]
    if "Feature Fraction" in df.columns and df["Feature Fraction"].nunique() > 1:
        group_cols.append("Feature Fraction")

    summary = _aggregate(df, group_cols)

    os.makedirs(AGG_DIR, exist_ok=True)
    out_path = os.path.join(AGG_DIR, f"{method}_aggregated.csv")
    summary.to_csv(out_path, index=False)
    print(f"[{method}] wrote {out_path}  ({len(summary)} rows, "
          f"{df['RunID'].nunique()} runs)")
    return summary


def _format_block(method, summary):
    if summary is None:
        return f"\n=== {method.upper()} ===\n  (no runs found)\n"

    lines = [f"\n=== {method.upper()} (aggregated over runs) ===\n"]
    for _, row in summary.iterrows():
        header = f"  Cancer: {row['Cancer']}  Method: {row['Method']}"
        if "Feature Fraction" in row:
            header += f"  Fraction: {row['Feature Fraction']}"
        lines.append(header)
        for metric in METRIC_COLS:
            mean = row.get(f"{metric}_mean")
            std = row.get(f"{metric}_std")
            mn = row.get(f"{metric}_min")
            mx = row.get(f"{metric}_max")
            if pd.isna(std):
                std = 0.0
            lines.append(
                f"    {metric:<14} mean={mean:.4f}  std={std:.4f}  "
                f"min={mn:.4f}  max={mx:.4f}"
            )
        lines.append(f"    n_runs        {int(row.get('n_runs', 0))}")
        lines.append("")
    return "\n".join(lines)


def main():
    methods = sys.argv[1:] if len(sys.argv) > 1 else ["selectkbest"]

    summaries = {}
    for m in methods:
        summaries[m] = aggregate_method(m)

    os.makedirs(AGG_DIR, exist_ok=True)
    txt_path = os.path.join(AGG_DIR, "aggregated_summary.txt")
    with open(txt_path, "w") as f:
        f.write("# Aggregated Wrapper Method Results\n")
        for m in methods:
            f.write(_format_block(m, summaries[m]))
    print(f"\nWrote {txt_path}")


if __name__ == "__main__":
    main()

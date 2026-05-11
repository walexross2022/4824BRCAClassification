"""
Embedded/aggregate_results.py

Aggregate per-run CSVs produced by `run_parallel.py` into mean / std / min /
max summaries. Also merges with any pre-existing 10-run summaries from
the original scripts (e.g. LASSO BRCA or RF all datasets).

Reads:
    Embedded/results/runs/<method>/run_<NN>_<CANCER>.csv
    Embedded/results/<METHOD>/<CANCER>/<CANCER>_run_results.csv   (pre-existing)

Writes:
    Embedded/results/aggregated/<method>_aggregated.csv
    Embedded/results/aggregated/aggregated_summary.txt
"""

import glob
import os
import sys

import numpy as np
import pandas as pd

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(THIS_DIR, "results")
RUNS_DIR = os.path.join(RESULTS_DIR, "runs")
AGG_DIR = os.path.join(RESULTS_DIR, "aggregated")

METRIC_COLS = ["accuracy", "macro_f1", "weighted_f1",
               "macro_recall", "weighted_recall", "roc_auc"]


def _load_new_runs(method):
    """Load CSVs from run_parallel.py output."""
    paths = sorted(glob.glob(os.path.join(RUNS_DIR, method, "run_*.csv")))
    if not paths:
        return pd.DataFrame()
    frames = [pd.read_csv(p) for p in paths]
    return pd.concat(frames, ignore_index=True)


def _load_existing_runs(method):
    """Load pre-existing 10-run CSVs from the original script output."""
    method_dir = os.path.join(RESULTS_DIR, method.upper())
    if not os.path.isdir(method_dir):
        method_dir = os.path.join(RESULTS_DIR, method)
    if not os.path.isdir(method_dir):
        return pd.DataFrame()

    frames = []
    for cancer_dir in glob.glob(os.path.join(method_dir, "*")):
        if not os.path.isdir(cancer_dir):
            continue
        csv_path = glob.glob(os.path.join(cancer_dir, "*_run_results.csv"))
        for p in csv_path:
            df = pd.read_csv(p)
            # Add Cancer column if missing
            if "Cancer" not in df.columns:
                cancer_name = os.path.basename(cancer_dir)
                df["Cancer"] = cancer_name
            if "Method" not in df.columns:
                df["Method"] = method.upper()
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _normalize_cols(df, method):
    """Ensure consistent column naming between new and old run CSVs."""
    col_map = {
        "Accuracy": "accuracy",
        "Macro F1": "macro_f1",
        "Weighted F1": "weighted_f1",
        "ROC-AUC": "roc_auc",
        "Runtime (s)": "train_time",
    }
    df = df.rename(columns=col_map)
    # Ensure method name is consistent
    if "Method" in df.columns:
        df["Method"] = df["Method"].replace({
            "LASSO": "LASSO",
            "RandomForest": "RandomForest",
            "Random Forest": "RandomForest",
            "ANOVA / F-test": "ANOVA",
        })
    return df


def aggregate_method(method):
    new_df = _load_new_runs(method)
    old_df = _load_existing_runs(method)

    if new_df.empty and old_df.empty:
        print(f"[{method}] no run files found, skipping")
        return None

    new_df = _normalize_cols(new_df, method)
    old_df = _normalize_cols(old_df, method)

    # Merge: prefer new runs where they exist, keep old ones for cancers
    # that weren't re-run
    if not new_df.empty and not old_df.empty:
        # Get cancers covered by new runs
        new_cancers = new_df["Cancer"].unique()
        # Keep old runs only for cancers NOT in new runs
        old_df = old_df[~old_df["Cancer"].isin(new_cancers)]
        combined = pd.concat([new_df, old_df], ignore_index=True)
    elif new_df.empty:
        combined = old_df
    else:
        combined = new_df

    available_metrics = [c for c in METRIC_COLS if c in combined.columns]
    if "train_time" in combined.columns and "train_time" not in available_metrics:
        available_metrics.append("train_time")
    if "n_features" in combined.columns:
        available_metrics.append("n_features")

    group_cols = ["Cancer", "Method"]
    agg_funcs = ["mean", "std", "min", "max"]
    summary = combined.groupby(group_cols)[available_metrics].agg(agg_funcs)
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    n_runs = combined.groupby(group_cols).size().rename("n_runs")
    summary = summary.join(n_runs)

    os.makedirs(AGG_DIR, exist_ok=True)
    out_path = os.path.join(AGG_DIR, f"{method}_aggregated.csv")
    summary.reset_index().to_csv(out_path, index=False)
    print(f"[{method}] wrote {out_path}  ({len(summary)} rows)")
    return summary.reset_index()


def _format_block(method, summary):
    if summary is None:
        return f"\n=== {method.upper()} ===\n  (no runs found)\n"

    lines = [f"\n=== {method.upper()} (aggregated over runs) ===\n"]
    for _, row in summary.iterrows():
        lines.append(f"  Cancer: {row['Cancer']}  Method: {row['Method']}")
        for metric in METRIC_COLS:
            mean_key = f"{metric}_mean"
            std_key = f"{metric}_std"
            if mean_key not in row:
                continue
            mean = row[mean_key]
            std = row.get(std_key, 0.0)
            if pd.isna(std):
                std = 0.0
            lines.append(
                f"    {metric:<14} mean={mean:.4f}  std={std:.4f}")
        if "n_features_mean" in row:
            lines.append(
                f"    {'n_features':<14} mean={row['n_features_mean']:.1f}"
                f"  std={row.get('n_features_std', 0.0):.1f}")
        if "train_time_mean" in row:
            lines.append(
                f"    {'train_time':<14} mean={row['train_time_mean']:.2f}s"
                f"  std={row.get('train_time_std', 0.0):.2f}s")
        lines.append(f"    n_runs        {int(row.get('n_runs', 0))}")
        lines.append("")
    return "\n".join(lines)


def main():
    methods = sys.argv[1:] if len(sys.argv) > 1 else ["lasso", "rf"]

    summaries = {}
    for m in methods:
        summaries[m] = aggregate_method(m)

    os.makedirs(AGG_DIR, exist_ok=True)
    txt_path = os.path.join(AGG_DIR, "aggregated_summary.txt")
    with open(txt_path, "w") as f:
        f.write("# Aggregated Embedded Method Results\n")
        for m in methods:
            f.write(_format_block(m, summaries[m]))
    print(f"\nWrote {txt_path}")


if __name__ == "__main__":
    main()

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(BASE_DIR, "Baseline", "results")
os.makedirs(OUT_DIR, exist_ok=True)

CANCERS = ["BRCA", "COAD", "PRAD"]
METRICS = ["accuracy", "macro_f1", "weighted_f1", "macro_recall", "weighted_recall"]
METRIC_LABELS = ["Accuracy", "Macro F1", "Weighted F1", "Macro Recall", "Weighted Recall"]

# ---- Load all run results ----
all_data = {}
for cancer in CANCERS:
    csv_path = os.path.join(OUT_DIR, f"{cancer}_run_results.csv")
    if not os.path.exists(csv_path):
        print(f"  WARNING: {csv_path} not found, skipping {cancer}")
        continue
    df = pd.read_csv(csv_path)
    all_data[cancer] = df
    print(f"Loaded {cancer}: {len(df)} runs")

if not all_data:
    print("No run results found. Run the baseline scripts first.")
    exit(1)

# ---- 1. Side-by-side aggregated bar chart (mean +/- std) ----
n_cancers = len(all_data)
n_metrics = len(METRICS)
x = np.arange(n_metrics)
width = 0.8 / n_cancers

plt.figure(figsize=(14, 7))
colors_cancer = ["#4C72B0", "#55A868", "#C44E52"]

for i, (cancer, color) in enumerate(zip(all_data, colors_cancer)):
    df = all_data[cancer]
    means = [df[m].mean() for m in METRICS]
    stds  = [df[m].std()  for m in METRICS]
    offset = (i - (n_cancers - 1) / 2) * width
    bars = plt.bar(x + offset, means, width, yerr=stds, capsize=4,
                   label=cancer, color=color, edgecolor="black", linewidth=1.0)
    for bar, mean, std in zip(bars, means, stds):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 0.01,
                 f"{mean:.3f}", ha="center", va="bottom", fontsize=7, fontweight="bold", rotation=45)

plt.xticks(x, METRIC_LABELS, fontsize=11)
plt.ylabel("Score", fontsize=12)
plt.title("Baseline Logistic Regression - Multi-Cancer Comparison (mean +/- std over 10 runs)", fontsize=13, fontweight="bold")
plt.ylim(0, 1.25)
plt.legend(fontsize=11)
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "cross_cancer_aggregated_comparison.png"), dpi=150)
plt.close()
print("  Saved cross_cancer_aggregated_comparison.png")

# ---- 2. Per-metric grouped bar charts (one subplot per metric) ----
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes_flat = axes.flatten()

for idx, (metric, label) in enumerate(zip(METRICS, METRIC_LABELS)):
    ax = axes_flat[idx]
    cancer_names = list(all_data.keys())
    means = [all_data[c][metric].mean() for c in cancer_names]
    stds  = [all_data[c][metric].std()  for c in cancer_names]
    bars = ax.bar(cancer_names, means, yerr=stds, capsize=5,
                  color=colors_cancer[:len(cancer_names)], edgecolor="black", linewidth=1.2)
    for bar, mean, std in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 0.005,
                f"{mean:.4f}\n+/-{std:.4f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.set_title(label, fontsize=13, fontweight="bold")
    ax.set_ylim(0, 1.2)
    ax.grid(axis="y", alpha=0.3)

# Remove unused subplot
axes_flat[-1].axis("off")

fig.suptitle("Per-Metric Comparison Across Cancers (10 runs each)", fontsize=15, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "cross_cancer_per_metric_comparison.png"), dpi=150, bbox_inches="tight")
plt.close()
print("  Saved cross_cancer_per_metric_comparison.png")

# ---- 3. Combined summary table ----
summary_rows = []
for cancer in all_data:
    df = all_data[cancer]
    for metric, label in zip(METRICS, METRIC_LABELS):
        summary_rows.append({
            "Cancer": cancer,
            "Metric": label,
            "Mean":   f"{df[metric].mean():.4f}",
            "Std":    f"{df[metric].std():.4f}",
            "Min":    f"{df[metric].min():.4f}",
            "Max":    f"{df[metric].max():.4f}",
        })

summary_df = pd.DataFrame(summary_rows)

with open(os.path.join(OUT_DIR, "cross_cancer_summary.txt"), "w") as f:
    f.write("CROSS-CANCER BASELINE SUMMARY - 10 runs\n")
    for cancer in all_data:
        f.write(f"\n--- {cancer} ---\n")
        sub = summary_df[summary_df["Cancer"] == cancer][["Metric", "Mean", "Std", "Min", "Max"]]
        f.write(sub.to_string(index=False) + "\n")

print(f"\nSummary saved")
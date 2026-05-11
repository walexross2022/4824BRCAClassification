import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# Paths to all methods' run results
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
METHODS = {
    "Baseline": os.path.join(BASE_DIR, "Baseline", "results"),
    "ANOVA (Filter)": os.path.join(BASE_DIR, "Filter", "results"),
    "LASSO (Embedded)": os.path.join(BASE_DIR, "Embedded", "results", "LASSO"),
    "Random Forest (Embedded)": os.path.join(BASE_DIR, "Embedded", "results", "RandomForest"),
}

CANCERS = ["BRCA", "COAD", "PRAD"]
METRICS = ["accuracy", "macro_f1", "weighted_f1", "macro_recall", "weighted_recall", "roc_auc"]
METRIC_LABELS = ["Accuracy", "Macro F1", "Weighted F1", "Macro Recall", "Weighted Recall", "ROC-AUC"]

OUT_DIR = os.path.join(BASE_DIR, "Embedded", "results", "Comparison")
os.makedirs(OUT_DIR, exist_ok=True)

# ---- Load all method results ----
all_data = {}
for method, path in METHODS.items():
    method_data = {}
    for cancer in CANCERS:
        # LASSO and RF store per-cancer in subdirectories; baseline and filter directly in results
        csv_path = os.path.join(path, cancer, f"{cancer}_run_results.csv")
        if not os.path.exists(csv_path):
            csv_path = os.path.join(path, f"{cancer}_run_results.csv")
        if not os.path.exists(csv_path):
            print(f"  WARNING: {csv_path} not found, skipping {method}/{cancer}")
            continue
        df = pd.read_csv(csv_path)
        method_data[cancer] = df
    if method_data:
        all_data[method] = method_data
        print(f"Loaded {method}: {sorted(method_data.keys())}")

if not all_data:
    print("No run results found. Run the baseline/embedded/filter scripts first.")
    exit(1)

# ---- 1. Side-by-side aggregated bar chart (mean +/- std) across methods ----
n_methods = len(all_data)
n_metrics = len(METRICS)
x = np.arange(n_metrics)
width = 0.8 / n_methods

colors_methods = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2", "#64B5CD", "#FFB74D", "#A1887F"]

fig, ax = plt.subplots(figsize=(16, 8))

for i, method in enumerate(all_data):
    all_means = []
    all_stds = []
    for metric in METRICS:
        vals = []
        for cancer in CANCERS:
            if cancer in all_data[method]:
                vals.extend(all_data[method][cancer][metric].tolist())
        all_means.append(np.mean(vals))
        all_stds.append(np.std(vals))
    
    offset = (i - (n_methods - 1) / 2) * width
    bars = ax.bar(x + offset, all_means, width, yerr=all_stds, capsize=4,
                  label=method, color=colors_methods[i], edgecolor="black", linewidth=0.8)
    for bar, mean in zip(bars, all_means):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.015,
                f"{mean:.3f}", ha="center", va="bottom", fontsize=7, fontweight="bold", rotation=45)

ax.set_xticks(x)
ax.set_xticklabels(METRIC_LABELS, fontsize=11)
ax.set_ylabel("Score", fontsize=12)
ax.set_title("Multi-Method Comparison Across All Cancers (mean +/- std, aggregated over 10 runs)", fontsize=14, fontweight="bold")
ax.set_ylim(0, 1.25)
ax.legend(fontsize=10, ncol=2)
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "cross_method_aggregated_comparison.png"), dpi=150)
plt.close()
print("  Saved cross_method_aggregated_comparison.png")

# ---- 2. Per-metric grouped bar charts (one subplot per metric) ----
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes_flat = axes.flatten()

for idx, (metric, label) in enumerate(zip(METRICS, METRIC_LABELS)):
    ax = axes_flat[idx]
    method_names = list(all_data.keys())
    means = []
    stds = []
    for method in method_names:
        vals = []
        for cancer in CANCERS:
            if cancer in all_data[method]:
                vals.extend(all_data[method][cancer][metric].tolist())
        means.append(np.mean(vals))
        stds.append(np.std(vals))
    
    bars = ax.bar(method_names, means, yerr=stds, capsize=5,
                  color=colors_methods[:len(method_names)], edgecolor="black", linewidth=1.2)
    for bar, mean, std in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 0.005,
                f"{mean:.4f}\n+/-{std:.4f}", ha="center", va="bottom", fontsize=8, fontweight="bold")
    ax.set_title(label, fontsize=13, fontweight="bold")
    ax.set_ylim(0, 1.2)
    ax.grid(axis="y", alpha=0.3)
    ax.tick_params(axis="x", rotation=30)

for k in range(len(METRICS), len(axes_flat)):
    axes_flat[k].axis("off")

fig.suptitle("Per-Metric Comparison Across Methods (aggregated over all cancers, 10 runs each)", fontsize=15, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "cross_method_per_metric_comparison.png"), dpi=150, bbox_inches="tight")
plt.close()
print("  Saved cross_method_per_metric_comparison.png")

# ---- 3. Per-cancer comparison (separate plot per cancer) ----
for cancer in CANCERS:
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes_flat = axes.flatten()
    
    for idx, (metric, label) in enumerate(zip(METRICS, METRIC_LABELS)):
        ax = axes_flat[idx]
        method_names = []
        means = []
        stds = []
        for method in all_data:
            if cancer in all_data[method]:
                method_names.append(method)
                vals = all_data[method][cancer][metric].tolist()
                means.append(np.mean(vals))
                stds.append(np.std(vals))
        
        bars = ax.bar(method_names, means, yerr=stds, capsize=5,
                      color=colors_methods[:len(method_names)], edgecolor="black", linewidth=1.2)
        for bar, mean, std in zip(bars, means, stds):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 0.005,
                    f"{mean:.4f}\n+/-{std:.4f}", ha="center", va="bottom", fontsize=7, fontweight="bold")
        ax.set_title(label, fontsize=12, fontweight="bold")
        ax.set_ylim(0, 1.25)
        ax.grid(axis="y", alpha=0.3)
        ax.tick_params(axis="x", rotation=30)
    
    for k in range(len(METRICS), len(axes_flat)):
        axes_flat[k].axis("off")
    
    fig.suptitle(f"{cancer} - Per-Metric Method Comparison (10 runs each)", fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f"{cancer}_method_comparison.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {cancer}_method_comparison.png")

# ---- 4. Combined summary table ----
summary_rows = []
for method in all_data:
    for cancer in CANCERS:
        if cancer not in all_data[method]:
            continue
        df = all_data[method][cancer]
        for metric, label in zip(METRICS, METRIC_LABELS):
            summary_rows.append({
                "Method": method,
                "Cancer": cancer,
                "Metric": label,
                "Mean": f"{df[metric].mean():.4f}",
                "Std": f"{df[metric].std():.4f}",
                "Min": f"{df[metric].min():.4f}",
                "Max": f"{df[metric].max():.4f}",
            })

summary_df = pd.DataFrame(summary_rows)

with open(os.path.join(OUT_DIR, "cross_method_summary.txt"), "w") as f:
    f.write("CROSS-METHOD FEATURE SELECTION SUMMARY - 10 runs per method\n")
    for method in all_data:
        f.write(f"\n{'='*60}\n")
        f.write(f"  {method}\n")
        f.write(f"{'='*60}\n")
        for cancer in CANCERS:
            if cancer not in all_data[method]:
                continue
            f.write(f"\n  --- {cancer} ---\n")
            sub = summary_df[(summary_df["Method"] == method) & (summary_df["Cancer"] == cancer)][["Metric", "Mean", "Std", "Min", "Max"]]
            f.write("  " + sub.to_string(index=False).replace("\n", "\n  ") + "\n")

print(f"\nSummary saved to cross_method_summary.txt")

# ---- 5. Feature counts comparison ----
fig, ax = plt.subplots(figsize=(12, 6))
x = np.arange(len(CANCERS))
width = 0.8 / n_methods

for i, method in enumerate(all_data):
    means = []
    for cancer in CANCERS:
        if cancer in all_data[method]:
            means.append(all_data[method][cancer]["n_features"].mean())
        else:
            means.append(0)
    offset = (i - (n_methods - 1) / 2) * width
    bars = ax.bar(x + offset, means, width, label=method, color=colors_methods[i], edgecolor="black", linewidth=0.8)
    for bar, mean in zip(bars, means):
        if mean > 0:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 500,
                    f"{mean:.0f}", ha="center", va="bottom", fontsize=7, fontweight="bold", rotation=45)

ax.set_xticks(x)
ax.set_xticklabels(CANCERS, fontsize=12)
ax.set_ylabel("Number of Selected Features", fontsize=12)
ax.set_title("Feature Count Comparison Across Methods", fontsize=14, fontweight="bold")
ax.legend(fontsize=10, ncol=2)
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "feature_count_comparison.png"), dpi=150)
plt.close()
print("  Saved feature_count_comparison.png")

print("\n[DONE] All comparison plots generated.")
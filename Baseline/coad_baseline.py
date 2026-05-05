import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, recall_score, classification_report, confusion_matrix

# Configuration
import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(BASE_DIR, "Baseline", "results")
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(OUT_DIR, exist_ok=True)

CANCER = "COAD"
N_RUNS = 10

# 1. Load expression matrix
print(f"[{CANCER}] Loading expression matrix...")
expr_path = os.path.join(DATA_DIR, f"TCGA-{CANCER}", "coad_expression_matrix.csv")
expr = pd.read_csv(expr_path, index_col=0)
print(f"  Expression shape: {expr.shape}")

# 2. Load subtypes
sub_path = os.path.join(DATA_DIR, f"TCGA-{CANCER}", "coad_subtypes.csv")
sub = pd.read_csv(sub_path)
sub.columns = sub.columns.str.strip()
sub = sub[["pan.samplesID", "Subtype_Selected"]].dropna()
sub["sample_id"] = sub["pan.samplesID"].str[:12]

# 3. Align sample IDs (deduplicate by averaging)
expr.index = expr.index.str[:12]
expr = expr.groupby(expr.index).mean()
common = expr.index.intersection(sub["sample_id"])
print(f"  Common samples: {len(common)}")

X = expr.loc[common].values
y = sub.set_index("sample_id").loc[common, "Subtype_Selected"]

print(f"  Class distribution:\n{y.value_counts()}")

all_records = []
sum_cm = None

for run in range(N_RUNS):
    seed = np.random.randint(0, 2**31 - 1)
    print(f"\n[{CANCER}] Run {run+1}/{N_RUNS} (seed={seed})")

    # 4. Train / test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=seed
    )

    # 5. Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 6. Train Logistic Regression
    model = LogisticRegression(max_iter=1000, random_state=seed)
    model.fit(X_train_scaled, y_train)

    # 7. Evaluate
    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average="macro")
    f1_weighted = f1_score(y_test, y_pred, average="weighted")
    rec_macro = recall_score(y_test, y_pred, average="macro")
    rec_weighted = recall_score(y_test, y_pred, average="weighted")

    print(f"    Accuracy:    {acc:.4f} | Macro F1: {f1_macro:.4f} | W F1: {f1_weighted:.4f} | Macro Rec: {rec_macro:.4f} | W Rec: {rec_weighted:.4f}")

    cm = confusion_matrix(y_test, y_pred)
    if sum_cm is None:
        sum_cm = cm
    else:
        sum_cm += cm

    all_records.append({
        "run": run + 1,
        "seed": seed,
        "accuracy": acc,
        "macro_f1": f1_macro,
        "weighted_f1": f1_weighted,
        "macro_recall": rec_macro,
        "weighted_recall": rec_weighted,
    })

results_df = pd.DataFrame(all_records)
results_df.to_csv(f"{OUT_DIR}/{CANCER}_run_results.csv", index=False)

print(f"\n[{CANCER}] Summary over {N_RUNS} runs:")
print(results_df.describe())

# ---- Aggregated bar chart ----
plt.figure(figsize=(10, 6))
metrics_agg = ["Accuracy", "Macro F1", "Weighted F1", "Macro Recall", "Weighted Recall"]
means = [results_df["accuracy"].mean(), results_df["macro_f1"].mean(),
         results_df["weighted_f1"].mean(), results_df["macro_recall"].mean(),
         results_df["weighted_recall"].mean()]
stds  = [results_df["accuracy"].std(),  results_df["macro_f1"].std(),
         results_df["weighted_f1"].std(), results_df["macro_recall"].std(),
         results_df["weighted_recall"].std()]
colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B2", "#CCB974"]
xpos = np.arange(len(metrics_agg))
bars = plt.bar(xpos, means, yerr=stds, capsize=5, color=colors, edgecolor="black", linewidth=1.2)
for i, (mean, std) in enumerate(zip(means, stds)):
    plt.text(i, mean + std + 0.01, f"{mean:.3f}\n+/-{std:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
plt.xticks(xpos, metrics_agg)
plt.ylim(0, 1.2)
plt.title(f"{CANCER} - Baseline LR Over {N_RUNS} Runs (mean +/- std)")
plt.ylabel("Score")
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/{CANCER}_aggregated_metrics.png", dpi=150)
plt.close()
print(f"  Saved aggregated metrics chart.")

# ---- Aggregated confusion matrix ----
plt.figure(figsize=(6, 5))
sns.heatmap(sum_cm, annot=True, fmt="d", cmap="Blues", cbar=False,
            xticklabels=sorted(y.unique()), yticklabels=sorted(y.unique()))
plt.title(f"{CANCER} - Aggregated Confusion Matrix ({N_RUNS} Runs)")
plt.ylabel("True Label")
plt.xlabel("Predicted Label")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/{CANCER}_aggregated_confusion_matrix.png", dpi=150)
plt.close()
print(f"  Saved aggregated confusion matrix plot.")

# ---- Per-run line plot ----
plt.figure(figsize=(12, 6))
for col, label, color in zip(
    ["accuracy", "macro_f1", "weighted_f1", "macro_recall", "weighted_recall"],
    metrics_agg, colors,
):
    plt.plot(results_df["run"], results_df[col], marker="o", label=label, color=color)
plt.xlabel("Run")
plt.ylabel("Score")
plt.title(f"{CANCER} - Per-Run Metrics Over {N_RUNS} Runs")
plt.xticks(results_df["run"])
plt.ylim(0, 1.1)
plt.legend()
plt.grid(axis="both", alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/{CANCER}_per_run_metrics.png", dpi=150)
plt.close()
print(f"  Saved per-run metrics chart.")

# ---- Text summary ----
with open(f"{OUT_DIR}/{CANCER}_summary.txt", "w") as f:
    f.write(f"{CANCER} Baseline Logistic Regression - {N_RUNS} Runs\n\n")
    f.write(f"{'Metric':<20} {'Mean':>10} {'Std':>10} {'Min':>10} {'Max':>10}\n")
    f.write("-" * 60 + "\n")
    for col, label in zip(
        ["accuracy", "macro_f1", "weighted_f1", "macro_recall", "weighted_recall"],
        ["Accuracy", "Macro F1", "Weighted F1", "Macro Recall", "Weighted Recall"],
    ):
        f.write(f"{label:<20} {results_df[col].mean():>10.4f} {results_df[col].std():>10.4f} {results_df[col].min():>10.4f} {results_df[col].max():>10.4f}\n")
    f.write(f"\nPer-run details:\n")
    f.write(results_df.to_string(index=False))

print(f"[{CANCER}] Done - run results saved to {OUT_DIR}/")

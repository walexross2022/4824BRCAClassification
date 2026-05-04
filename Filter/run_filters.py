# Filter/run_filters.py

import os
import subprocess
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# -----------------------------
# Configuration
# -----------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FILTER_DIR = os.path.join(BASE_DIR, "Filter")
RESULTS_DIR = os.path.join(FILTER_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

SCRIPTS = [
    "variance_filter.py",
    "anova_filter.py",
    "mutual_info_filter.py"
]

RESULT_FILES = [
    "variance_filter_results.csv",
    "anova_filter_results.csv",
    "mutual_info_filter_results.csv"
]

print("Running all filter methods...\n")

# -----------------------------
# STEP 1: Run All Filter Scripts
# -----------------------------
for script in SCRIPTS:
    print(f"Running {script}...")
    subprocess.run(["py", script], cwd=FILTER_DIR, check=True)
    print(f"Finished {script}\n")

# -----------------------------
# STEP 2: Load and Combine Results
# -----------------------------
print("Combining filter results...")

dfs = []
for file in RESULT_FILES:
    path = os.path.join(RESULTS_DIR, file)
    df = pd.read_csv(path)
    dfs.append(df)

combined = pd.concat(dfs, ignore_index=True)
combined.to_csv(os.path.join(RESULTS_DIR, "filter_comparison.csv"), index=False)

print("Saved combined results to filter_comparison.csv")

# -----------------------------
# STEP 3: Comparison Plot
# -----------------------------
plt.figure(figsize=(10, 6))

x = combined["Method"]
acc = combined["Accuracy"]
f1 = combined["Macro F1"]
runtime = combined["Runtime (s)"]

plt.plot(x, acc, marker="o", linewidth=2, label="Accuracy")
plt.plot(x, f1, marker="o", linewidth=2, label="Macro F1")
plt.plot(x, runtime / runtime.max(), marker="o", linewidth=2,
         label="Runtime (normalized)")

plt.title("Filter Method Comparison")
plt.ylabel("Score")
plt.ylim(0, 1.05)
plt.grid(axis="y", alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "filter_comparison_plot.png"), dpi=150)
plt.close()

print("Saved comparison plot to filter_comparison_plot.png")

# -----------------------------
# STEP 4: Save Summary TXT
# -----------------------------
with open(os.path.join(RESULTS_DIR, "filter_summary.txt"), "w") as f:
    f.write("=== Filter Method Comparison Summary ===\n\n")
    f.write(combined.to_string(index=False))
    f.write("\n\nBest Accuracy: ")
    f.write(combined.loc[combined["Accuracy"].idxmax(), "Method"])
    f.write("\nBest Macro F1: ")
    f.write(combined.loc[combined["Macro F1"].idxmax(), "Method"])
    f.write("\nFastest Runtime: ")
    f.write(combined.loc[combined["Runtime (s)"].idxmin(), "Method"])

print("Saved summary to filter_summary.txt")
print("\nAll filter methods complete.")
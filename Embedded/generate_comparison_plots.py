import matplotlib.pyplot as plt
import numpy as np
import os

# Create results directory for plots
output_dir = "4824BRCAClassification/Embedded/results/Comparison"
os.makedirs(output_dir, exist_ok=True)

datasets = ['BRCA', 'COAD', 'PRAD']
methods = ['Baseline', 'ANOVA (Filter)', 'SelectKBest (Wrapper)', 'LASSO (Embedded)', 'Random Forest (Embedded)']

# Accuracy Data from Tables 4, 6, and 7
# Baseline values are from the "Results" text section in your draft
accuracy_data = {
    'Baseline': [0.8412, 0.8544, 0.7920],
    'ANOVA (Filter)': [0.8328, 0.8447, 0.7800],
    'SelectKBest (Wrapper)': [0.8085, 0.8155, 0.6100],
    'LASSO (Embedded)': [0.7671, 0.7681, 0.7612],
    'Random Forest (Embedded)': [0.8128, 0.8696, 0.7612]
}

# Macro F1 Data from Tables 4, 6, and 7
f1_data = {
    'Baseline': [0.7820, 0.7122, 0.4215],
    'ANOVA (Filter)': [0.7745, 0.6058, 0.3719],
    'SelectKBest (Wrapper)': [0.7151, 0.6915, 0.3728],
    'LASSO (Embedded)': [0.6779, 0.5627, 0.3717],
    'Random Forest (Embedded)': [0.6303, 0.5979, 0.2982]
}

# Feature Counts from Tables 4, 6, and 7
feature_counts = {
    'Baseline': [60660, 60660, 60660],
    'ANOVA (Filter)': [5000, 5000, 5000],
    'SelectKBest (Wrapper)': [30330, 30330, 30330],
    'LASSO (Embedded)': [44708, 38574, 39029],
    'Random Forest (Embedded)': [7894, 1742, 2345]
}

def create_grouped_bar(data, title, ylabel, filename, ylim=None):
    x = np.arange(len(datasets))
    width = 0.15
    multiplier = 0

    fig, ax = plt.subplots(figsize=(10, 6), layout='constrained')

    for attribute, measurement in data.items():
        offset = width * multiplier
        ax.bar(x + offset, measurement, width, label=attribute)
        multiplier += 1

    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_xticks(x + width * (len(methods)-1) / 2, datasets)
    ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1))
    if ylim:
        ax.set_ylim(ylim)
    
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig(os.path.join(output_dir, filename), dpi=300)
    plt.close()

# Plot 1: Accuracy Comparison
create_grouped_bar(accuracy_data, 'Classification Accuracy by Feature Selection Method', 'Accuracy', 'accuracy_comparison.png', ylim=(0, 1.0))

# Plot 2: Macro F1 Comparison
create_grouped_bar(f1_data, 'Macro F1-Score by Feature Selection Method', 'Macro F1', 'f1_comparison.png', ylim=(0, 1.0))

# Plot 3: Feature Reduction (Log Scale)
fig, ax = plt.subplots(figsize=(10, 6), layout='constrained')
x = np.arange(len(datasets))
width = 0.15
multiplier = 0

for attribute, measurement in feature_counts.items():
    offset = width * multiplier
    ax.bar(x + offset, measurement, width, label=attribute)
    multiplier += 1

ax.set_yscale('log')
ax.set_title('Feature Count Reduction (Log Scale)', fontsize=14, fontweight='bold')
ax.set_ylabel('Number of Selected Features', fontsize=12)
ax.set_xticks(x + width * (len(methods)-1) / 2, datasets)
ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1))
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.savefig(os.path.join(output_dir, 'feature_reduction.png'), dpi=300)
plt.close()

print(f"Plots saved to {output_dir}")
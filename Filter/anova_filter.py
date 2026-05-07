# Filter/anova_filter.py
import warnings
warnings.filterwarnings("ignore")

import os
import sys
import pandas as pd

# Add parent directory to path to import shared_utils and app
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import X, y
from sklearn.feature_selection import SelectKBest, f_classif
from shared_utils import evaluate_feature_set

# Configuration
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(BASE_DIR, "Filter", "results")
os.makedirs(OUT_DIR, exist_ok=True)

CANCER = "BRCA"
TOP_K = 5000

print(f"\n=== Starting ANOVA Filter for {CANCER} ===")

# -----------------------------
# STEP 1: ANOVA Feature Selection
# -----------------------------
# We keep the implementation the same: selection on the full dataset
# as in the original script.
selector = SelectKBest(score_func=f_classif, k=TOP_K)
selector.fit(X, y)
selected_genes = X.columns[selector.get_support()]
X_selected = X[selected_genes]

# -----------------------------
# STEP 2: Standardized Evaluation
# -----------------------------
labels = sorted(y.unique())
results = evaluate_feature_set(
    X_selected, y, "ANOVA", CANCER, labels, OUT_DIR
)

print(f"\n[{CANCER}] ANOVA filter refactored and evaluated.")

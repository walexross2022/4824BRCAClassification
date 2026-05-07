# Wrapper/selectkbest_wrapper.py
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
OUT_DIR = os.path.join(BASE_DIR, "Wrapper", "results")
os.makedirs(OUT_DIR, exist_ok=True)

CANCER = "BRCA"
# Keeping one representative fraction for the main comparison, 
# but could be looped if multiple were needed.
FEATURE_FRACTION = 0.10 

print(f"\n=== Starting SelectKBest Wrapper for {CANCER} ===")

total_features = X.shape[1]
n_features_to_select = max(1, int(total_features * FEATURE_FRACTION))

# -----------------------------
# STEP 1: Feature Selection
# -----------------------------
selector = SelectKBest(f_classif, k=n_features_to_select)
selector.fit(X, y)
selected_genes = X.columns[selector.get_support()]
X_selected = X[selected_genes]

# -----------------------------
# STEP 2: Standardized Evaluation
# -----------------------------
labels = sorted(y.unique())
results = evaluate_feature_set(
    X_selected, y, "SelectKBest", CANCER, labels, OUT_DIR
)

print(f"\n[{CANCER}] SelectKBest wrapper refactored and evaluated.")

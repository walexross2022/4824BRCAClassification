import warnings
warnings.filterwarnings("ignore")

import os
import sys
import pandas as pd
import numpy as np

# Add parent directory to path to import shared_utils and data_utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Embedded.data_utils import load_tcga_data
from shared_utils import evaluate_feature_set

# Configuration
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(BASE_DIR, "Baseline", "results", "BRCA")
os.makedirs(OUT_DIR, exist_ok=True)

CANCER = "BRCA"

print(f"[{CANCER}] Loading data via standardized loader...")
X, y, labels = load_tcga_data(CANCER)

# -----------------------------
# Standardized Evaluation
# -----------------------------
# For Baseline, we use ALL features (no selection)
results = evaluate_feature_set(
    X, y, "Baseline", CANCER, labels, OUT_DIR
)

print(f"[{CANCER}] Done - baseline results saved to {OUT_DIR}/")

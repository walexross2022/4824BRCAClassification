# Wrapper/app.py

import os
import pandas as pd

# -----------------------------
# Configuration
# -----------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")

CANCER = "BRCA"
PREFIX = "brca"

# -----------------------------
# STEP 1: Load Expression Matrix
# -----------------------------
print(f"[{CANCER}] Loading expression matrix...")
expr_path = os.path.join(DATA_DIR, f"TCGA-{CANCER}", f"{PREFIX}_expression_matrix.csv")
expr = pd.read_csv(expr_path, index_col=0)
print(f"  Expression shape: {expr.shape}")

# -----------------------------
# STEP 2: Load Subtypes
# -----------------------------
sub_path = os.path.join(DATA_DIR, f"TCGA-{CANCER}", f"{PREFIX}_subtypes.csv")
sub = pd.read_csv(sub_path)

sub.columns = sub.columns.str.strip()
sub = sub[["pan.samplesID", "Subtype_Selected"]].dropna()

# -----------------------------
# STEP 3: Align Sample IDs
# -----------------------------
sub["sample_id"] = sub["pan.samplesID"].str[:12]
sub = sub.drop_duplicates(subset="sample_id")

expr.index = expr.index.str[:12]
expr = expr.groupby(expr.index).mean()

common = expr.index.intersection(sub["sample_id"])
print(f"  Common samples: {len(common)}")

# -----------------------------
# STEP 4: Final Dataset
# -----------------------------
X = expr.loc[common]
y = sub.set_index("sample_id").loc[common, "Subtype_Selected"]

print(f"  Class distribution:\n{y.value_counts()}")
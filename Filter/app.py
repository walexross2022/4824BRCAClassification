# Filter/app.py

import os
import pandas as pd


# -----------------------------
# Configuration
# -----------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")

# Dataset-specific configuration
DATASET_CONFIG = {
    "BRCA": {
        "prefix": "brca",
        "label_col": "Subtype_Selected",
    },
    "COAD": {
        "prefix": "coad",
        "label_col": "Subtype_Selected",
    },
    "PRAD": {
        "prefix": "prad",
        "label_col": "Subtype_Selected",
    },
}


# -----------------------------
# Dataset Loader
# -----------------------------
def load_dataset(cancer_type="BRCA"):
    """
    Load and preprocess expression + subtype labels for a given cancer dataset.

    Parameters
    ----------
    cancer_type : str
        One of {"BRCA", "COAD", "PRAD"}.

    Returns
    -------
    X : pd.DataFrame
        Gene expression matrix (samples x genes)
    y : pd.Series
        Class labels aligned to X
    """
    cancer_type = cancer_type.upper()

    if cancer_type not in DATASET_CONFIG:
        raise ValueError(
            f"Unsupported cancer type '{cancer_type}'. "
            f"Choose from {list(DATASET_CONFIG.keys())}."
        )

    prefix = DATASET_CONFIG[cancer_type]["prefix"]
    label_col = DATASET_CONFIG[cancer_type]["label_col"]

    # -----------------------------
    # STEP 1: Load Expression Matrix
    # -----------------------------
    print(f"[{cancer_type}] Loading expression matrix...")
    expr_path = os.path.join(
        DATA_DIR,
        f"TCGA-{cancer_type}",
        f"{prefix}_expression_matrix.csv"
    )
    expr = pd.read_csv(expr_path, index_col=0)
    print(f"  Expression shape: {expr.shape}")

    # -----------------------------
    # STEP 2: Load Subtypes
    # -----------------------------
    sub_path = os.path.join(
        DATA_DIR,
        f"TCGA-{cancer_type}",
        f"{prefix}_subtypes.csv"
    )
    sub = pd.read_csv(sub_path)

    sub.columns = sub.columns.str.strip()

    required_cols = ["pan.samplesID", label_col]
    missing = [col for col in required_cols if col not in sub.columns]
    if missing:
        raise ValueError(
            f"[{cancer_type}] Missing required columns in subtype file: {missing}"
        )

    sub = sub[required_cols].dropna()

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
    y = sub.set_index("sample_id").loc[common, label_col]

    print(f"  Final shape: {X.shape}")
    print(f"  Class distribution:\n{y.value_counts()}\n")

    return X, y


# -----------------------------
# Optional local test
# -----------------------------
if __name__ == "__main__":
    for cancer in ["BRCA", "COAD", "PRAD"]:
        X, y = load_dataset(cancer)
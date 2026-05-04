import pandas as pd
import numpy as np
import os

DATA_DIR = "data"

def load_tcga_data(cancer_type):
    """
    Standardized data loader for TCGA datasets.
    Handles ID alignment, duplicate averaging, and subtype extraction.
    """
    prefix = cancer_type.lower()
    cancer_folder = f"TCGA-{cancer_type.upper()}"
    
    expr_path = os.path.join(DATA_DIR, cancer_folder, f"{prefix}_expression_matrix.csv")
    sub_path = os.path.join(DATA_DIR, cancer_folder, f"{prefix}_subtypes.csv")
    
    if not os.path.exists(expr_path) or not os.path.exists(sub_path):
        raise FileNotFoundError(f"Data files for {cancer_type} not found.")

    # Load data
    expr = pd.read_csv(expr_path, index_col=0)
    sub = pd.read_csv(sub_path)
    
    # Clean subtype labels
    sub.columns = sub.columns.str.strip()
    sub = sub[["pan.samplesID", "Subtype_Selected"]].dropna()
    sub["sample_id"] = sub["pan.samplesID"].str[:12]
    sub = sub.drop_duplicates(subset="sample_id")
    
    # Align expression data
    expr.index = expr.index.str[:12]
    expr = expr.groupby(expr.index).mean()
    
    # Find common samples
    common = expr.index.intersection(sub["sample_id"])
    X = expr.loc[common]
    y = sub.set_index("sample_id").loc[common, "Subtype_Selected"]
    
    return X, y, sorted(y.unique())

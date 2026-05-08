# Wrapper/forwardselection_wrapper.py
import os
import time
import warnings

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

warnings.filterwarnings("ignore")

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from app import X, y

try:
    import cupy as cp  # type: ignore[reportMissingImports]
    from cuml.linear_model import LogisticRegression  # type: ignore[reportMissingImports]
    from cuml.preprocessing import StandardScaler  # type: ignore[reportMissingImports]
except ImportError as exc:
    raise ImportError(
        "This wrapper requires RAPIDS cuML and CuPy. Run it inside a WSL2/Linux "
        "RAPIDS environment with cupy and cuml installed."
    ) from exc

from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder

# -----------------------------
# Configuration
# -----------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(BASE_DIR, "Wrapper", "results")
os.makedirs(OUT_DIR, exist_ok=True)

CANCER = "BRCA"
RANDOM_STATE = 42
FEATURE_FRACTIONS = [0.2]
TOP_VARIANCE_FEATURES = 1000
CV_FOLDS = 2
LR_MAX_ITER = 500
LR_TOL = 1e-4
LR_C = 1.0


def to_numpy(values):
    if hasattr(values, "to_cupy"):
        values = values.to_cupy()
    if isinstance(values, cp.ndarray):
        return cp.asnumpy(values)
    return np.asarray(values)


def make_logistic_regression():
    return LogisticRegression(max_iter=LR_MAX_ITER, tol=LR_TOL, C=LR_C)


def make_cv_folds(y_train_encoded):
    splitter = StratifiedKFold(
        n_splits=CV_FOLDS,
        shuffle=True,
        random_state=RANDOM_STATE
    )
    folds = []

    for train_idx, valid_idx in splitter.split(
        np.zeros_like(y_train_encoded),
        y_train_encoded
    ):
        folds.append((
            cp.asarray(train_idx, dtype=cp.int32),
            cp.asarray(valid_idx, dtype=cp.int32),
            y_train_encoded[valid_idx]
        ))

    return folds


def score_feature_subset(X_train_scaled, y_train_gpu, folds, feature_indices):
    scores = []
    feature_indices_gpu = cp.asarray(feature_indices, dtype=cp.int32)

    for train_idx_gpu, valid_idx_gpu, y_valid_cpu in folds:
        model = make_logistic_regression()
        X_fold_train = X_train_scaled[train_idx_gpu][:, feature_indices_gpu]
        X_fold_valid = X_train_scaled[valid_idx_gpu][:, feature_indices_gpu]
        y_fold_train = y_train_gpu[train_idx_gpu]

        model.fit(X_fold_train, y_fold_train)
        y_pred = to_numpy(model.predict(X_fold_valid)).astype(y_valid_cpu.dtype)
        scores.append(f1_score(y_valid_cpu, y_pred, average="macro"))

    return float(np.mean(scores))


def run_forward_selection(X_train_scaled, y_train_gpu, y_train_encoded,
                          n_features_to_select):
    selected = []
    remaining = list(range(X_train_scaled.shape[1]))
    folds = make_cv_folds(y_train_encoded)

    for step in range(n_features_to_select):
        best_feature = None
        best_score = -np.inf

        for candidate in remaining:
            candidate_features = selected + [candidate]
            score = score_feature_subset(
                X_train_scaled,
                y_train_gpu,
                folds,
                candidate_features
            )

            if score > best_score:
                best_feature = candidate
                best_score = score

        selected.append(best_feature)
        remaining.remove(best_feature)

        print(
            f"  Step {step + 1:>3}/{n_features_to_select}: "
            f"feature={best_feature} cv_macro_f1={best_score:.4f}"
        )

    return selected


# -----------------------------
# STEP 1: Train / Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.30,
    stratify=y,
    random_state=RANDOM_STATE
)

print(f"\nTrain shape: {X_train.shape}")
print(f"Test shape:  {X_test.shape}")

total_features = X_train.shape[1]

# -----------------------------
# STEP 2: GPU Variance Pre-filter
# -----------------------------
X_train_gpu_full = cp.asarray(X_train.to_numpy(dtype=np.float32, copy=True))
X_test_gpu_full = cp.asarray(X_test.to_numpy(dtype=np.float32, copy=True))

top_k = min(TOP_VARIANCE_FEATURES, total_features)
variances_gpu = cp.var(X_train_gpu_full, axis=0)
top_feature_indices_gpu = cp.argsort(variances_gpu)[-top_k:][::-1]
top_feature_indices = cp.asnumpy(top_feature_indices_gpu).astype(int)
top_feature_names = X_train.columns[top_feature_indices]

X_train_gpu = X_train_gpu_full[:, top_feature_indices_gpu]
X_test_gpu = X_test_gpu_full[:, top_feature_indices_gpu]

print(f"Features after GPU variance pre-filter: {X_train_gpu.shape[1]}")

label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train).astype(np.int32)
y_test_encoded = label_encoder.transform(y_test).astype(np.int32)
y_train_gpu = cp.asarray(y_train_encoded)

# -----------------------------
# STEP 3: Scale on GPU
# -----------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_gpu)
X_test_scaled = scaler.transform(X_test_gpu)

# -----------------------------
# STEP 4: Run Forward Selection Sweep
# -----------------------------
all_results = []

for fraction in FEATURE_FRACTIONS:
    n_features_to_select = max(1, int(X_train_scaled.shape[1] * fraction))
    print("\n" + "=" * 50)
    print(f"Running RAPIDS Forward Selection: fraction={fraction}  "
          f"n_features={n_features_to_select}")
    print("=" * 50)

    start_time = time.time()

    selected_feature_indices = run_forward_selection(
        X_train_scaled,
        y_train_gpu,
        y_train_encoded,
        n_features_to_select
    )

    selected_feature_indices_gpu = cp.asarray(
        selected_feature_indices,
        dtype=cp.int32
    )
    X_train_selected = X_train_scaled[:, selected_feature_indices_gpu]
    X_test_selected = X_test_scaled[:, selected_feature_indices_gpu]
    selected_features = X_train_selected.shape[1]

    print(f"Original features:  {total_features}")
    print(f"Selected features:  {selected_features}")
    print("Starting LogisticRegression training on GPU...")

    train_start = time.time()
    model = make_logistic_regression()
    model.fit(X_train_selected, y_train_gpu)
    train_time = time.time() - train_start

    print(f"Training finished in {train_time:.2f} seconds")

    y_pred = to_numpy(model.predict(X_test_selected)).astype(y_test_encoded.dtype)

    acc = accuracy_score(y_test_encoded, y_pred)
    f1_macro = f1_score(y_test_encoded, y_pred, average="macro")
    f1_weighted = f1_score(y_test_encoded, y_pred, average="weighted")

    runtime = time.time() - start_time

    print(f"Accuracy:      {acc:.4f}")
    print(f"Macro F1:      {f1_macro:.4f}")
    print(f"Weighted F1:   {f1_weighted:.4f}")
    print(f"Runtime (s):   {runtime:.2f}")

    all_results.append({
        "Method": "RAPIDS Forward Selection",
        "Feature Fraction": fraction,
        "Original Features": total_features,
        "Selected Features": selected_features,
        "Accuracy": acc,
        "Macro F1": f1_macro,
        "Weighted F1": f1_weighted,
        "Runtime (s)": runtime,
        "Selected Feature Names": ";".join(
            map(str, top_feature_names[selected_feature_indices])
        )
    })

# -----------------------------
# STEP 5: Save Results Table
# -----------------------------
results = pd.DataFrame(all_results)
results.to_csv(os.path.join(OUT_DIR, "forward_selection_results.csv"), index=False)

# -----------------------------
# STEP 6: Save Feature Count Comparison Plot
# -----------------------------
plt.figure(figsize=(10, 6))

plt.plot(results["Selected Features"], results["Accuracy"],
         marker="o", linewidth=2, label="Accuracy")
plt.plot(results["Selected Features"], results["Macro F1"],
         marker="o", linewidth=2, label="Macro F1")
plt.plot(results["Selected Features"], results["Weighted F1"],
         marker="o", linewidth=2, label="Weighted F1")

plt.title(f"{CANCER} - RAPIDS Forward Selection Sweep")
plt.xlabel("Number of Selected Features")
plt.ylabel("Score")
plt.ylim(0, 1.05)
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "forward_selection_sweep.png"), dpi=150)
plt.close()

# -----------------------------
# STEP 7: Save Feature Count Plot
# -----------------------------
plt.figure(figsize=(10, 6))

plt.plot(results["Feature Fraction"], results["Selected Features"],
         marker="o", linewidth=2)

plt.title(f"{CANCER} - RAPIDS Selected Features vs Feature Fraction")
plt.xlabel("Feature Fraction")
plt.ylabel("Selected Features")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "forward_selected_features.png"), dpi=150)
plt.close()

# -----------------------------
# STEP 8: Save Summary TXT
# -----------------------------
with open(os.path.join(OUT_DIR, "forward_selection_results.txt"), "w") as f:
    f.write(f"=== {CANCER} RAPIDS Forward Selection Sweep ===\n\n")
    f.write(results.to_string(index=False))
    f.write("\n\nBest Accuracy (n_features): ")
    f.write(str(results.loc[results["Accuracy"].idxmax(), "Selected Features"]))
    f.write("\nBest Macro F1 (n_features): ")
    f.write(str(results.loc[results["Macro F1"].idxmax(), "Selected Features"]))
    f.write("\nFastest Runtime (n_features): ")
    f.write(str(results.loc[results["Runtime (s)"].idxmin(), "Selected Features"]))

print(f"\n[{CANCER}] Done - RAPIDS forward selection sweep saved to {OUT_DIR}/")
# feature_selection_lasso.py

import warnings
warnings.filterwarnings("ignore")

from app import X, y

import numpy as np
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

from sklearn.feature_selection import VarianceThreshold

selector = VarianceThreshold(threshold=0.01)
X = selector.fit_transform(X)

print("After variance filter:", X.shape)

# STEP 1: Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3,
    stratify=y,
    random_state=42
)

print("Train shape:", X_train.shape)
print("Test shape:", X_test.shape)

# STEP 2: Scaling
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# STEP 3: LASSO Experiments
print("\nRunning LASSO experiments...")

for C in [0.1, 0.05, 0.01]:

    print(f"\n--- Testing C = {C} ---")

    start_time = time.time()

    lasso = LogisticRegression(
        penalty='l1',
        solver='saga',
        max_iter=500,
        C=C,
        verbose=1 
    )

    lasso.fit(X_train, y_train)

    # Feature selection
    selected = np.sum(lasso.coef_ != 0)

    # Predictions
    y_pred = lasso.predict(X_test)

    # Metrics
    accuracy = lasso.score(X_test, y_test)
    f1 = f1_score(y_test, y_pred, average='macro')

    end_time = time.time()

    # Output
    print("Selected features:", selected)
    print("Accuracy:", accuracy)
    print("Macro F1:", f1)
    print("Runtime (seconds):", round(end_time - start_time, 2))

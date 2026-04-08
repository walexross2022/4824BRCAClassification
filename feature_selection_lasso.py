# feature_selection_lasso.py

import warnings
warnings.filterwarnings("ignore")

from app import X, y

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

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

from sklearn.feature_selection import VarianceThreshold

selector = VarianceThreshold(threshold=0.01)
X = selector.fit_transform(X) 

# STEP 3: LASSO Model
lasso = LogisticRegression(
    penalty='l1',
    solver='saga',
    max_iter=500,
    C=0.1,          # VERY IMPORTANT (stronger regularization)
    verbose=1       # shows progress
)

print("\nTraining LASSO model...")
lasso.fit(X_train, y_train)

# STEP 4: Feature Selection
# Count non-zero coefficients
selected_features = np.sum(lasso.coef_ != 0)

print("\nSelected features:", selected_features)


# STEP 5: Evaluation
y_pred = lasso.predict(X_test)

accuracy = lasso.score(X_test, y_test)
f1 = f1_score(y_test, y_pred, average='macro')

print("\nModel Performance:")
print("Accuracy:", accuracy)
print("Macro F1:", f1)

#Results
#max_iter reached after 577 seconds
#Selected features: 8413
#Model Performance:
#Accuracy: 0.89375
#Macro F1: 0.8778980162280018

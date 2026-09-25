import os
import json
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, roc_auc_score
from xgboost import XGBClassifier


# 1. Load Credit dataset

DATA_PATH = "./dataset/credit/credit.csv"

df = pd.read_csv(DATA_PATH)

train_raw, test_raw = train_test_split(
    df,
    test_size=0.2,
    random_state=42
)

target = train_raw.columns[-1]
feature_cols = list(train_raw.columns[:-1])

print("Dataset shape :", df.shape)
print("Train shape   :", train_raw.shape)
print("Test shape    :", test_raw.shape)
print("Target        :", target)


# 2. Pearson ordering -- exactly the repository logic

pearson_df = df.copy()

# Same categorical conversion used for Pearson ordering
object_cols = pearson_df.columns[pearson_df.dtypes == "object"]

pearson_df[object_cols] = pearson_df[object_cols].apply(
    lambda x: pd.Categorical(x).codes
)

corr = pearson_df.corr(method="pearson")[target]

corr = corr.drop(target)

order = corr.abs().sort_values(ascending=False).index.tolist()

print("\n=== MOST feature order ===")

for i, col in enumerate(order, 1):
    print(
        f"{i:2d}. {col:<25} | "
        f"|corr| = {corr[col]:.6f}"
    )


# 3. Training shift values


shift_values = {}

for col in feature_cols:

    if train_raw[col].dtype == "object":
        shift_values[col] = train_raw[col].mode()[0]

    else:
        shift_values[col] = train_raw[col].mean()

print("\n=== Training shift values ===")

for col in order:
    print(f"{col:<25}: {shift_values[col]}")


# 4. Exact XGB.py-style encoding
#
# IMPORTANT:
# Each dataframe is encoded independently.


def encode_like_xgb_py(data):

    data = data.copy()

    object_cols = data.columns[data.dtypes == "object"]

    data[object_cols] = data[object_cols].apply(
        lambda x: pd.Categorical(x).codes
    )

    return data


# 5. Prepare training set

train_encoded = encode_like_xgb_py(train_raw)


# 6. Official XGBoost default configuration

current_dir = os.path.abspath(".")

config_path = os.path.join(
    current_dir,
    "configs",
    "default",
    "xgboost.json"
)

with open(config_path, "r") as f:
    param_grid = json.load(f)

# Compatibility with sklearn 1.5.2
param_grid["eval_metric"] = ["logloss"]

print("\n=== Parameter grid ===")
print(param_grid)

# 7. GridSearch -- same as XGB.py

model = XGBClassifier()

grid_search = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    cv=5
)

grid_search.fit(
    train_encoded.iloc[:, :-1],
    train_encoded.iloc[:, -1]
)

best_params = grid_search.best_params_

print("\n=== Best parameters ===")
print(best_params)

print("\n=== Best CV score ===")
print(grid_search.best_score_)


# 8. Train downstream model

downstream = XGBClassifier(**best_params)

downstream.fit(
    train_encoded.iloc[:, :-1],
    train_encoded.iloc[:, -1]
)


# 9. Baseline test

baseline_test = encode_like_xgb_py(test_raw)

X_test = baseline_test.iloc[:, :-1]
y_test = baseline_test.iloc[:, -1]

y_pred = downstream.predict(X_test)
y_prob = downstream.predict_proba(X_test)[:, 1]

baseline_acc = accuracy_score(y_test, y_pred)
baseline_auc = roc_auc_score(y_test, y_prob)

print("\n=== BASELINE ===")
print(f"Accuracy : {baseline_acc:.10f}")
print(f"ROC-AUC  : {baseline_auc:.10f}")


# 10. MOST feature-shift experiment
# One model, independent test copy for every degree.

results = []

n_features = len(order)

print("\n" + "=" * 105)
print("Credit | XGBoost | MOST feature shift")
print("=" * 105)

print(
    f"{'degree':>8}"
    f"{'features':>10}"
    f"{'Accuracy':>15}"
    f"{'Delta Acc.':>16}"
    f"{'ROC-AUC':>15}"
    f"{'Delta AUC':>16}"
)

for k in range(n_features + 1):

    degree = k / n_features

    # Fresh independent copy
    shifted_test = test_raw.copy(deep=True)

    # Shift first k features according to MOST order
    shifted_features = order[:k]

    for col in shifted_features:
        shifted_test[col] = shift_values[col]

    # IMPORTANT:
    # Encode this test dataframe independently,
    # exactly as XGB.py does.
    shifted_encoded = encode_like_xgb_py(shifted_test)

    X_shifted = shifted_encoded.iloc[:, :-1]
    y_shifted = shifted_encoded.iloc[:, -1]

    pred = downstream.predict(X_shifted)
    prob = downstream.predict_proba(X_shifted)[:, 1]

    acc = accuracy_score(y_shifted, pred)
    auc = roc_auc_score(y_shifted, prob)

    delta_acc = (acc - baseline_acc) / baseline_acc
    delta_auc = (auc - baseline_auc) / baseline_auc

    results.append({
        "degree": degree,
        "features": k,
        "accuracy": acc,
        "delta_accuracy": delta_acc,
        "roc_auc": auc,
        "delta_auc": delta_auc
    })

    print(
        f"{degree:8.3f}"
        f"{k:10d}"
        f"{acc:15.10f}"
        f"{delta_acc:16.10f}"
        f"{auc:15.10f}"
        f"{delta_auc:16.10f}"
    )


# 11. Save results

results_df = pd.DataFrame(results)

results_df.to_csv(
    "credit_xgb_most_exact.csv",
    index=False
)

print("\nSaved:")
print("credit_xgb_most_exact.csv")
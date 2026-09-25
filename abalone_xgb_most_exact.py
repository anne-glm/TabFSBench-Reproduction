import json
import os
import importlib.util

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor


# 1. Paths

ROOT = os.path.dirname(os.path.abspath(__file__))

DATA_PATH = os.path.join(
    ROOT, "dataset", "abalone", "abalone.csv"
)

CONFIG_PATH = os.path.join(
    ROOT, "configs", "default", "xgboost.json"
)

# 2. Load dataset

df = pd.read_csv(DATA_PATH)

print("=" * 70)
print("DATASET")
print("=" * 70)
print("Shape:", df.shape)
print("Columns:", list(df.columns))
print()

# 3. Train / test split
#    Exactly as official run_experiment.py

train_raw, test_raw = train_test_split(
    df,
    test_size=0.2,
    random_state=42
)

train_raw = train_raw.copy(deep=True)
test_raw = test_raw.copy(deep=True)

print("Train shape:", train_raw.shape)
print("Test shape :", test_raw.shape)
print()



# 4. Pearson ordering
# Official methodology:
# object columns are converted to categorical codes
# before calculating Pearson correlation.

pearson_df = df.copy(deep=True)

object_cols = pearson_df.columns[
    pearson_df.dtypes == "object"
]

for col in object_cols:
    pearson_df[col] = pd.Categorical(
        pearson_df[col]
    ).codes

corr = pearson_df.corr(method="pearson")

target_col = df.columns[-1]

target_corr = corr[target_col].drop(
    labels=[target_col]
)

ordered_features = (
    target_corr.abs()
    .sort_values(ascending=False)
    .index
    .tolist()
)


print("=" * 70)
print("MOST FEATURE ORDER")
print("=" * 70)

for i, feature in enumerate(ordered_features, start=1):
    print(
        f"{i:2d}. {feature:20s} "
        f"corr={target_corr[feature]: .6f} "
        f"|corr|={abs(target_corr[feature]): .6f}"
    )

print()


# 5. Calculate training shift values
# Numerical -> training mean
# Categorical -> training mode

shift_values = {}

for col in ordered_features:

    if train_raw[col].dtype == "object":
        shift_values[col] = train_raw[col].mode(
            dropna=True
        ).iloc[0]
    else:
        shift_values[col] = train_raw[col].mean()


print("=" * 70)
print("SHIFT VALUES")
print("=" * 70)

for feature in ordered_features:
    print(
        f"{feature:20s} -> {shift_values[feature]}"
    )

print()

# 6. Load official XGBoost default configuration


with open(CONFIG_PATH, "r") as f:
    param_grid = json.load(f)

# Compatibility adjustment:
# sklearn 1.5.2 expects grid values to be lists.
param_grid["eval_metric"] = ["rmse"]

print("=" * 70)
print("PARAMETER GRID")
print("=" * 70)
print(param_grid)
print()


# 7. Encode TRAIN exactly according to XGB.py


train_set = train_raw.copy(deep=True)

train_object_cols = train_set.columns[
    train_set.dtypes == "object"
]

for col in train_object_cols:
    train_set[col] = pd.Categorical(
        train_set[col]
    ).codes


# 8. GridSearchCV

model = XGBRegressor()

grid_search = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    cv=5
)

print("=" * 70)
print("RUNNING GRID SEARCH")
print("=" * 70)

grid_search.fit(
    train_set.iloc[:, :-1],
    train_set.iloc[:, -1]
)

best_params = grid_search.best_params_

print("Best params:")
print(best_params)

print()
print("Best CV score:", grid_search.best_score_)
print()


# 9. Train downstream model ONCE

downstream = XGBRegressor(**best_params)

downstream.fit(
    train_set.iloc[:, :-1],
    train_set.iloc[:, -1]
)


# 10. Function for official-style independent test encoding


def encode_test_exact(test_set):

    test_encoded = test_set.copy(deep=True)

    object_cols = test_encoded.columns[
        test_encoded.dtypes == "object"
    ]

    for col in object_cols:
        test_encoded[col] = pd.Categorical(
            test_encoded[col]
        ).codes

    return test_encoded


# 11. Evaluate MOST degrees


n_features = len(ordered_features)

results = []

baseline_test = test_raw.copy(deep=True)

baseline_encoded = encode_test_exact(
    baseline_test
)

X_test = baseline_encoded.iloc[:, :-1]
y_test = baseline_encoded.iloc[:, -1]

baseline_pred = downstream.predict(X_test)

baseline_rmse = np.sqrt(
    mean_squared_error(
        y_test,
        baseline_pred
    )
)

print("=" * 70)
print("BASELINE")
print("=" * 70)
print(f"RMSE = {baseline_rmse:.10f}")
print()

# 12. Degree 0 -> 1


for k in range(n_features + 1):

    shifted_test = test_raw.copy(deep=True)

    # Shift the first k MOST relevant features
    for feature in ordered_features[:k]:
        shifted_test[feature] = shift_values[feature]

    # IMPORTANT:
    # Each test set is independently categorical-encoded,
    # matching official XGB.py behavior.
    shifted_encoded = encode_test_exact(
        shifted_test
    )

    X_shifted = shifted_encoded.iloc[:, :-1]
    y_shifted = shifted_encoded.iloc[:, -1]

    predictions = downstream.predict(
        X_shifted
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_shifted,
            predictions
        )
    )

    delta = (
        (rmse - baseline_rmse)
        / baseline_rmse
    )

    degree = k / n_features

    results.append({
        "degree": degree,
        "shifted_features": k,
        "rmse": rmse,
        "delta": delta
    })

    print(
        f"degree={degree:.3f} | "
        f"features={k:2d} | "
        f"RMSE={rmse:.10f} | "
        f"Delta={delta:.10f}"
    )


# 13. Save results

results_df = pd.DataFrame(results)

OUTPUT_PATH = os.path.join(
    ROOT,
    "abalone_xgb_most_exact.csv"
)

results_df.to_csv(
    OUTPUT_PATH,
    index=False
)

print()
print("=" * 70)
print("DONE")
print("=" * 70)
print("Saved:", OUTPUT_PATH)
print()

print(results_df.to_string(index=False))
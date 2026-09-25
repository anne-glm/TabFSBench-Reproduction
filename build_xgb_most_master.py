import os
import pandas as pd


ROOT = os.path.dirname(os.path.abspath(__file__))


FILES = {
    "Credit": "credit_xgb_most_exact.csv",
    "Abalone": "abalone_xgb_most_exact.csv",
    "Electricity": "electricity_xgb_most_exact.csv",
    "Concrete": "concrete_xgb_most_exact.csv",
}


all_results = []


for dataset, filename in FILES.items():

    path = os.path.join(ROOT, filename)

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Missing file: {path}"
        )

    df = pd.read_csv(path)

    df.insert(
        0,
        "dataset",
        dataset
    )

    all_results.append(df)


master = pd.concat(
    all_results,
    ignore_index=True
)


# Save complete master table

output_path = os.path.join(
    ROOT,
    "xgb_most_master.csv"
)

master.to_csv(
    output_path,
    index=False
)


print("=" * 80)
print("XGBOOST + MOST — MASTER TABLE")
print("=" * 80)
print()

print(master.to_string(index=False))

print()
print("=" * 80)
print("SAVED")
print("=" * 80)
print(output_path)


# Baseline table


baseline = master[
    master["degree"] == 0
].copy()

print()
print("=" * 80)
print("BASELINES")
print("=" * 80)
print()

print(
    baseline.to_string(index=False)
)


# Maximum-shift table

maximum = master[
    master["degree"] == 1
].copy()

print()
print("=" * 80)
print("DEGREE = 1.0")
print("=" * 80)
print()

print(
    maximum.to_string(index=False)
)


# Save separate summary tables


baseline_path = os.path.join(
    ROOT,
    "xgb_most_baselines.csv"
)

maximum_path = os.path.join(
    ROOT,
    "xgb_most_degree1.csv"
)

baseline.to_csv(
    baseline_path,
    index=False
)

maximum.to_csv(
    maximum_path,
    index=False
)

print()
print("Additional files:")
print(baseline_path)
print(maximum_path)
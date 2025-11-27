"""
predict_fundamentals.py

Use the trained fundamentals models to predict future
revenue, net income, and operating cash flow for each company
for 1, 3, and 5 years ahead (based on the latest year in the dataset).

Output:
- predicted_fundamentals.csv
"""

import os
import json

import numpy as np
import pandas as pd
import joblib

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "stock_dataset.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")
OUTPUT_PATH = os.path.join(BASE_DIR, "predicted_fundamentals.csv")

PRED_HORIZONS = [1, 3, 5]
TARGETS = ["revenue", "net_income", "operating_cash_flow"]


def load_dataset():
    df = pd.read_csv(DATA_PATH)
    df = df.sort_values(["ticker", "year"]).reset_index(drop=True)
    return df


def load_feature_columns():
    feat_path = os.path.join(MODELS_DIR, "fundamentals_feature_columns.json")
    if not os.path.exists(feat_path):
        raise RuntimeError(
            f"Feature columns JSON not found at {feat_path}. "
            f"Run train_fundamentals.py first."
        )
    with open(feat_path, "r") as f:
        cols = json.load(f)
    return cols


def load_models():
    models = {}
    for target in TARGETS:
        models[target] = {}
        for h in PRED_HORIZONS:
            filename = f"fund_{target}_{h}y.pkl"
            path = os.path.join(MODELS_DIR, filename)
            if not os.path.exists(path):
                print(f"[WARN] Model file missing: {path} (skipping)")
                models[target][h] = None
            else:
                models[target][h] = joblib.load(path)
    return models


def main():
    df = load_dataset()
    feature_cols = load_feature_columns()
    models = load_models()

    # Latest row per ticker (this is our "base year" for predictions)
    latest = df.sort_values("year").groupby("ticker").tail(1).reset_index(drop=True)

    rows = []
    for _, row in latest.iterrows():
        ticker = row["ticker"]
        name = row.get("name", "")
        base_year = int(row["year"])

        X = row[feature_cols].fillna(0.0).values.reshape(1, -1)

        record = {
            "ticker": ticker,
            "name": name,
            "base_year": base_year,
        }

        for target in TARGETS:
            for h in PRED_HORIZONS:
                model = models.get(target, {}).get(h)
                key = f"pred_{target}_{h}y"
                if model is None:
                    record[key] = np.nan
                else:
                    pred = model.predict(X)[0]
                    record[key] = float(pred)

        rows.append(record)

    out_df = pd.DataFrame(rows)
    out_df.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved predicted fundamentals → {OUTPUT_PATH}")
    print(out_df.head())


if __name__ == "__main__":
    main()

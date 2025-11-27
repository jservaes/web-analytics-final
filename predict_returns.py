#predictions script! 
import pandas as pd
import joblib
import json
import os
import numpy as np


DATASET_PATH = "stock_dataset.csv"
MODELS_DIR = "models"
OUTPUT_CSV = "predicted_returns.csv"

TARGETS = ["1y", "2y", "3y", "4y", "5y"]


def load_features():
    """Load the feature list used during training."""
    path = os.path.join(MODELS_DIR, "feature_columns.json")
    with open(path, "r") as f:
        return json.load(f)


def load_models():
    """Load the trained multi-year models."""
    models = {}
    for t in TARGETS:
        model_path = os.path.join(MODELS_DIR, f"model_{t}_gb.pkl")
        models[t] = joblib.load(model_path)
    return models


def get_latest_rows(df):
    """
    For each ticker, keep only the most recent year available.
    Example: Apple 2024, NVIDIA 2024, etc.
    """
    return df.sort_values("year").groupby("ticker").tail(1)


def main():
    print("Loading dataset...")
    df = pd.read_csv(DATASET_PATH)

    # ★ match training preprocessing: replace inf with NaN
    df = df.replace([np.inf, -np.inf], np.nan)

    # Load features
    features = load_features()

    # Make sure features exist
    missing = [f for f in features if f not in df.columns]
    if missing:
        raise ValueError(f"Missing features in dataset: {missing}")

    # Get latest fundamentals per company
    latest = get_latest_rows(df)

    # ★ fill NaNs in feature columns with 0 
    latest[features] = latest[features].fillna(0)

    X = latest[features]

    models = load_models()

    # Prepare prediction table
    pred_df = latest[["ticker", "name", "year"]].reset_index(drop=True)

    for t in TARGETS:
        pred_df[f"pred_return_{t}"] = models[t].predict(X)

    pred_df.to_csv(OUTPUT_CSV, index=False)

    print(pred_df.to_string(index=False))


if __name__ == "__main__":
    main()

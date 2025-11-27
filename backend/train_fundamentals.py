"""
train_fundamentals.py

Train ML models to predict future fundamentals (revenue, net income,
operating cash flow) at 1, 3, and 5 years ahead.

Targets:
- target_revenue_1y, 3y, 5y
- target_net_income_1y, 3y, 5y
- target_ocf_1y, 3y, 5y   (operating_cash_flow)

Models:
- GradientBoostingRegressor

Outputs:
- models/fund_revenue_1y.pkl, ...
- models/fundamentals_feature_columns.json
- models/fundamentals_metrics.json
"""

import os
import json

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# BASE_DIR → the "final" folder (project root)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Now these point to the correct locations:
DATA_PATH = os.path.join(BASE_DIR, "stock_dataset.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")

# Make sure models folder exists
os.makedirs(MODELS_DIR, exist_ok=True)


PRED_HORIZONS = [1, 3, 5]
TARGETS = ["revenue", "net_income", "operating_cash_flow"]
SPLIT_YEAR = 2021  # same idea as for returns: earlier years for train, later for test


def load_dataset():
    df = pd.read_csv(DATA_PATH)
    # Ensure sorted by ticker/year
    df = df.sort_values(["ticker", "year"]).reset_index(drop=True)
    return df


def add_future_targets(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each target and horizon, add a column like target_revenue_1y
    which is the revenue at year + horizon for the SAME ticker.
    """
    df = df.copy()
    for target in TARGETS:
        if target not in df.columns:
            print(f"[WARN] target column '{target}' not in dataset; skipping.")
            continue
        for h in PRED_HORIZONS:
            col_name = f"target_{target}_{h}y"
            df[col_name] = (
                df.groupby("ticker")[target]
                .shift(-h)  # future value at t + h
            )
    return df


def get_feature_columns(df: pd.DataFrame):
    """
    Use numeric columns as features, excluding:
      - year (optional)
      - any 'return_' columns (future stock return targets)
      - any 'target_' columns (future fundamental targets)
    """
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    feature_cols = [
        c
        for c in num_cols
        if not c.startswith("return_")
        and not c.startswith("target_")
        and c != "year"  # you can include 'year' if you want time trend explicitly
    ]

    print(f"Using {len(feature_cols)} feature columns for fundamentals models.")
    return feature_cols


def train_one_model(df: pd.DataFrame, feature_cols, target_col: str):
    """
    Train/test split by SPLIT_YEAR, then fit a GradientBoostingRegressor.
    Returns fitted model and metrics dict.
    """
    # Drop rows with NaN target
    sub = df.dropna(subset=[target_col]).copy()
    if sub.empty:
        print(f"[WARN] No rows with non-NaN target for {target_col}; skipping.")
        return None, None

    # Features + target
    X = sub[feature_cols].fillna(0.0)
    y = sub[target_col].values

    # Train/test split by year
    train_mask = sub["year"] <= SPLIT_YEAR
    test_mask = sub["year"] > SPLIT_YEAR

    if train_mask.sum() == 0 or test_mask.sum() == 0:
        print(f"[WARN] Not enough data to split for {target_col}; skipping.")
        return None, None

    X_train, X_test = X[train_mask], X[test_mask]
    y_train, y_test = y[train_mask], y[test_mask]

    model = GradientBoostingRegressor(
        n_estimators=400,
        learning_rate=0.05,
        max_depth=3,
        random_state=42,
    )
    model.fit(X_train, y_train)

    # Evaluate
    def compute_metrics(y_true, y_pred):
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = float(np.sqrt(mse))
        r2 = r2_score(y_true, y_pred)
        return {"MAE": float(mae), "RMSE": rmse, "R2": float(r2)}

    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    metrics = {
        "train": compute_metrics(y_train, train_pred),
        "test": compute_metrics(y_test, test_pred),
        "n_train": int(train_mask.sum()),
        "n_test": int(test_mask.sum()),
    }

    return model, metrics


def main():
    df = load_dataset()
    df = add_future_targets(df)
    feature_cols = get_feature_columns(df)

    # Save feature columns for later prediction
    feat_path = os.path.join(MODELS_DIR, "fundamentals_feature_columns.json")
    with open(feat_path, "w") as f:
        json.dump(feature_cols, f, indent=2)

    all_metrics = {}

    for target in TARGETS:
        if target not in df.columns:
            continue
        all_metrics[target] = {}
        for h in PRED_HORIZONS:
            target_col = f"target_{target}_{h}y"
            print(f"\n=== Training {target_col} ===")
            model, metrics = train_one_model(df, feature_cols, target_col)

            if model is None:
                all_metrics[target][f"{h}y"] = None
                continue

            # Save model
            model_filename = f"fund_{target}_{h}y.pkl"
            model_path = os.path.join(MODELS_DIR, model_filename)

            import joblib
            joblib.dump(model, model_path)
            print(f"Saved model → {model_path}")

            all_metrics[target][f"{h}y"] = metrics
            print("Metrics:", metrics)

    # Save metrics
    metrics_path = os.path.join(MODELS_DIR, "fundamentals_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\nSaved fundamentals metrics → {metrics_path}")


if __name__ == "__main__":
    main()

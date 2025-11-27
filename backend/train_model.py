# method that learns the patterns (trains the ML model)
# loads dataset from stock_dataset.csv, trains regression model to predict stock returns, saves trained model, features and metrics 
# outputs into models/ folder 
# model_1y_xgb ->  file containing trained model, contains all the learned patterns. This is important 
# because we wont need to retrain model everytime it runs, and it will powwer /api/predict
# metrics_1y.json -> stores performance metrics of the trained model on training and test datasets. 
# Useful for monitoring model performance over time.
# feature_columns.json -> stores the exact set of features used during training.
# ensures the backend is giving the model the same exact features it was trained on and prevents mismatches 
import json
from pathlib import Path
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "stock_dataset.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)


# Columns that should NEVER be used as numeric features
ID_COLS = ["ticker", "cik", "name", "year"]

# Targets per horizon
TARGET_MAP = {
    "1y": "return_1y",
    "2y": "return_2y",
    "3y": "return_3y",
    "4y": "return_4y",
    "5y": "return_5y",
}


def load_dataset():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Dataset not found at {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    df = df.replace([np.inf, -np.inf], np.nan)

    target_cols = list(TARGET_MAP.values())
    feature_cols = [c for c in df.columns if c not in ID_COLS + target_cols]

    # Fill NaNs in features (model can't handle NaNs directly)
    df[feature_cols] = df[feature_cols].fillna(0)

    return df, feature_cols


def time_split(df: pd.DataFrame, target_col: str):
    """Time-based train/test split for a given target."""
    df_t = df.dropna(subset=[target_col]).copy()
    if df_t.empty:
        raise ValueError(f"No data available for target {target_col}")

    # Use 70% quantile of year as split point so it's not hardcoded
    split_year = int(df_t["year"].quantile(0.7))
    print(f"  Using split_year={split_year} for target {target_col}")

    train_df = df_t[df_t["year"] <= split_year].copy()
    test_df = df_t[df_t["year"] > split_year].copy()

    if train_df.empty or test_df.empty:
        raise ValueError(f"Train or test set empty for target {target_col}")

    return train_df, test_df


def train_one_horizon(h_label: str, df: pd.DataFrame, feature_cols):
    target_col = TARGET_MAP[h_label]
    print(f"\n=== Training horizon {h_label} ({target_col}) ===")

    train_df, test_df = time_split(df, target_col)

    X_train = train_df[feature_cols]
    y_train = train_df[target_col]
    X_test = test_df[feature_cols]
    y_test = test_df[target_col]

    model = GradientBoostingRegressor(
        n_estimators=300,
        max_depth=3,
        learning_rate=0.05,
        random_state=42,
    )
    model.fit(X_train, y_train)

    def compute_metrics(y_true, y_pred):
        mae = mean_absolute_error(y_true, y_pred)
        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        r2 = r2_score(y_true, y_pred)
        return {"MAE": float(mae), "RMSE": rmse, "R2": float(r2)}

    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    train_metrics = compute_metrics(y_train, train_pred)
    test_metrics = compute_metrics(y_test, test_pred)

    print("  Train:", train_metrics)
    print("  Test :", test_metrics)

    model_path = os.path.join(MODELS_DIR, f"model_{h_label}_gb.pkl")
    joblib.dump(model, model_path)
    print(f"  Saved model to {model_path}")

    return {"train": train_metrics, "test": test_metrics}


def main():
    print(f"Loading dataset from {DATA_PATH} ...")
    df, feature_cols = load_dataset()
    print(f"Rows: {len(df)}, Columns: {len(df.columns)}")
    print(f"Feature columns ({len(feature_cols)}): {feature_cols}")

    # Save feature list (used later when loading models)
    feature_path = os.path.join(MODELS_DIR, "feature_columns.json")
    with open(feature_path, "w") as f:
        json.dump(feature_cols, f, indent=2)
    print(f"Saved feature column list → {feature_path}")

    all_metrics = {}
    for h in ["1y", "2y", "3y", "4y", "5y"]:
        try:
            all_metrics[h] = train_one_horizon(h, df, feature_cols)
        except ValueError as e:
            print(f"Skipping horizon {h} due to error: {e}")

    metrics_path = os.path.join(MODELS_DIR, "metrics_multi_year.json")
    with open(metrics_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\nSaved metrics for all horizons → {metrics_path}")


if __name__ == "__main__":
    main()

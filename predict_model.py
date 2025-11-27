# helper that loads saved ML model, feature list and outputs predicted 1 year stock return
#important to power api/predict end point 

import json
from pathlib import Path

import joblib
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"

MODEL_PATH = MODEL_DIR / "model_1y_xgb.pkl"
FEATURES_PATH = MODEL_DIR / "feature_columns.json"


class StockReturnModel:
    def __init__(self):
        self.model = joblib.load(MODEL_PATH)
        with open(FEATURES_PATH) as f:
            self.feature_cols = json.load(f)

    def predict_from_features(self, feat_dict: dict) -> float:
        """
        feat_dict: {feature_name: value, ...} for ONE row.
        Must contain at least all feature_cols from training.
        """
        df = pd.DataFrame([feat_dict])
        df = df[self.feature_cols]  # ensure correct order/columns
        pred = self.model.predict(df)[0]
        return float(pred)

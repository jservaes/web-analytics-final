# backend/api.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import json
import pandas as pd
import joblib

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "stock_dataset.csv"
MODELS_DIR = BASE_DIR / "models"

app = FastAPI(title="Stock Analytics API")

# Allow frontend (static HTML/JS) to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- Load data and models once on startup ----------

# Main dataset (historical fundamentals + returns)
df = pd.read_csv(DATA_PATH)

# Feature columns used for return models
with open(MODELS_DIR / "feature_columns.json", "r") as f:
    RETURN_FEATURE_COLS = json.load(f)

# Feature columns used for fundamentals models (if present)
fund_feature_path = MODELS_DIR / "fundamentals_feature_columns.json"
if fund_feature_path.exists():
    with open(fund_feature_path, "r") as f:
        FUND_FEATURE_COLS = json.load(f)
else:
    FUND_FEATURE_COLS = RETURN_FEATURE_COLS  # fallback

# Load return models: 1y–5y
RETURN_MODELS = {}
for h in ["1y", "2y", "3y", "4y", "5y"]:
    path = MODELS_DIR / f"model_{h}_gb.pkl"
    if path.exists():
        RETURN_MODELS[h] = joblib.load(path)

# Load fundamentals models: 1-year ahead revenue / net income / operating cash flow
FUND_MODELS = {}
for target in ["revenue", "net_income", "operating_cash_flow"]:
    path = MODELS_DIR / f"fund_{target}_1y.pkl"
    if path.exists():
        FUND_MODELS[target] = joblib.load(path)


@app.get("/api/company-summary")
def company_summary(ticker: str):
    """
    Return:
      - history of fundamentals and price
      - predicted 1–5 year returns
      - predicted next-year revenue, net income, operating cash flow
    """
    t = ticker.upper()
    sub = df[df["ticker"] == t].sort_values("year")

    if sub.empty:
        raise HTTPException(status_code=404, detail=f"Ticker {t} not found in dataset")

    latest = sub.iloc[-1]

    # ---- Historical data (for plots) ----
    history = {
        "year": sub["year"].tolist(),
        "mean_price": sub["mean_price"].tolist(),
        "revenue": sub["revenue"].tolist(),
        "net_income": sub["net_income"].tolist(),
        "operating_cash_flow": sub["operating_cash_flow"].tolist(),
    }

    # ---- Build feature vector for predictions (from latest year) ----
    X_returns = latest[RETURN_FEATURE_COLS].fillna(0).to_frame().T

    returns_pred = {}
    for h, model in RETURN_MODELS.items():
        returns_pred[h] = float(model.predict(X_returns)[0])

    # Fundamentals predictions (only if models exist)
    fund_pred = {}
    if FUND_MODELS:
        X_fund = latest[FUND_FEATURE_COLS].fillna(0).to_frame().T
        for target, model in FUND_MODELS.items():
            fund_pred[target] = float(model.predict(X_fund)[0])

    return {
        "ticker": t,
        "name": latest["name"],
        "latest_year": int(latest["year"]),
        "history": history,
        "predictions": {
            "returns": returns_pred,  # 1y–5y expected returns
            "fundamentals_1y": {
                "base_year": int(latest["year"]),
                "predicted_year": int(latest["year"] + 1),
                "values": fund_pred,   # revenue / net_income / operating_cash_flow
            },
        },
    }

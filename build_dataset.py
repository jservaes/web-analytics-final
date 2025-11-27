# turns json from getData.py into clean tabular CSV with features and labels
# loops over companies and years, pulls out key metrics and calculates ratios
# outputs stock_dataset.csv 
import json
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
INPUT_JSON = BASE_DIR / "stockInfo" / "us_companies.json"
OUTPUT_CSV = BASE_DIR / "stock_dataset.csv"


def load_companies():
    if not INPUT_JSON.exists():
        raise FileNotFoundError(
            f"JSON not found at {INPUT_JSON}. "
            "Use the last good us_companies.json from your teammate / earlier run."
        )
    with open(INPUT_JSON, "r") as f:
        return json.load(f)


def build_rows(companies):
    rows = []

    for ticker, comp in companies.items():
        # keys like 'cik', 'name', '2012', '2013', ...
        cik = comp.get("cik")
        name = comp.get("name", ticker)

        year_keys = sorted(
            [y for y in comp.keys() if y.isdigit()],
            key=int
        )
        if len(year_keys) < 2:
            continue

        for i, year in enumerate(year_keys):
            yblock = comp[year].get("yearly", {})
            sp = yblock.get("stock_price") or {}
            price_t = sp.get("mean_price")
            if price_t is None or price_t <= 0:
                continue

            # helper to get future-year return over horizon h (1..5)
            def future_return(h):
                idx = i + h
                if idx < len(year_keys):
                    fut_year = year_keys[idx]
                    fut_block = comp[fut_year].get("yearly", {})
                    fut_sp = fut_block.get("stock_price") or {}
                    price_f = fut_sp.get("mean_price")
                    if price_f is not None and price_f > 0:
                        return (price_f - price_t) / price_t
                return None

            r_1y = future_return(1)
            r_2y = future_return(2)
            r_3y = future_return(3)
            r_4y = future_return(4)
            r_5y = future_return(5)

            # fundamentals for year t
            revenue = yblock.get("revenue")
            net_income = yblock.get("net_income")
            assets = yblock.get("assets")
            equity = yblock.get("equity")
            debt = yblock.get("debt")
            ocf = yblock.get("operating_cash_flow")

            # require basic accounting info
            if any(v is None for v in [revenue, net_income, assets, equity]):
                continue

            # ratios
            net_margin = net_income / revenue if revenue not in (None, 0) else None
            roa = net_income / assets if assets not in (None, 0) else None
            roe = net_income / equity if equity not in (None, 0) else None
            leverage = (
                debt / equity
                if (debt is not None and equity not in (None, 0))
                else None
            )
            ocf_margin = (
                ocf / revenue
                if (ocf is not None and revenue not in (None, 0))
                else None
            )

            # growth vs previous year
            rev_growth = None
            ni_growth = None
            if i > 0:
                prev_year = year_keys[i - 1]
                prev_block = comp[prev_year].get("yearly", {})
                prev_rev = prev_block.get("revenue")
                prev_ni = prev_block.get("net_income")
                if prev_rev not in (None, 0):
                    rev_growth = (revenue - prev_rev) / prev_rev
                if prev_ni not in (None, 0):
                    ni_growth = (net_income - prev_ni) / prev_ni

            rows.append({
                "ticker": ticker,
                "cik": cik,
                "name": name,
                "year": int(year),
                "mean_price": price_t,
                # multi-year targets
                "return_1y": r_1y,
                "return_2y": r_2y,
                "return_3y": r_3y,
                "return_4y": r_4y,
                "return_5y": r_5y,
                # fundamentals
                "revenue": revenue,
                "net_income": net_income,
                "assets": assets,
                "equity": equity,
                "debt": debt,
                "operating_cash_flow": ocf,
                # ratios
                "net_margin": net_margin,
                "roa": roa,
                "roe": roe,
                "leverage": leverage,
                "ocf_margin": ocf_margin,
                # growth
                "rev_growth": rev_growth,
                "ni_growth": ni_growth,
            })

    df = pd.DataFrame(rows)
    df = df.replace([np.inf, -np.inf], np.nan)

    # require at least 1-year target (others can be NaN)
    df = df.dropna(subset=["return_1y"])

    return df


def main():
    companies = load_companies()
    df = build_rows(companies)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved ML dataset: {OUTPUT_CSV}")
    print(df.head())
    print(f"Rows: {len(df)}, Columns: {len(df.columns)}")


if __name__ == "__main__":
    main()

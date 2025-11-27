import requests
import json
import os
from progress.bar import Bar
import time
from datetime import datetime
import statistics

import yfinance as yf

# ====== SEC + GAAP setup ======

HEADERS = {
    # Replace with your real contact email per SEC guidelines
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) your_email@example.com"
}

# Income statement – revenue
revenueTags = [
    "RevenuesNetOfInterestExpense",
    "Revenues",
    "RevenueFromContractWithCustomerExcludingAssessedTax",
    "RevenueFromContractWithCustomerIncludingAssessedTax",
    "SalesRevenueNet",
    "SalesRevenueGoodsNet",
    "SalesRevenueServicesNet",
    "SalesRevenueNetOfReturnsAndAllowances",
    "TotalRevenuesAndOtherIncome",
]

# Income statement – net income
netIncomeTags = [
    "NetIncomeLoss",
    "ProfitLoss",
    "NetIncomeLossAttributableToParent",
    "NetIncomeLossIncludingPortionAttributableToNoncontrollingInterest",
    "NetIncomeLossAvailableToCommonStockholdersBasic",
    "NetIncomeLossAvailableToCommonStockholdersDiluted",
]

# Balance sheet – assets
assetsTags = [
    "Assets",
    "AssetsNet",
]

# Balance sheet – equity
equityTags = [
    "StockholdersEquity",
    "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
    "StockholdersEquityValue",
    "PartnersCapital",
    "MembersEquity",
]

# Cash flow – operating cash flow
operatingCashFlowTags = [
    "NetCashProvidedByUsedInOperatingActivities",
    "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations",
    "NetCashProvidedByUsedInOperatingActivitiesDiscontinuedOperations",
    "NetCashProvidedByUsedInOperatingActivitiesOfDiscontinuedOperations",
]

# Balance sheet – long-term / noncurrent debt
debtLongTags = [
    "LongTermDebt",
    "LongTermDebtNoncurrent",
    "LongTermDebtAndCapitalLeaseObligations",
    "LongTermDebtAndCapitalLeaseObligationsNoncurrent",
    "DebtNoncurrent",
]


# ====== Helpers for tickers / files ======

def loadTickers():
    resp = requests.get("https://www.sec.gov/files/company_tickers.json",
                        headers=HEADERS)
    resp.raise_for_status()
    return resp.json()


def addTrailingZeros(num: int) -> str:
    s = str(num)
    return s.zfill(10)


def retrieveCompanyData(cik: str):
    url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
    resp = requests.get(url, headers=HEADERS)
    if resp.status_code != 200:
        print(f"  ! Failed for CIK {cik}: {resp.status_code}")
        return None
    return resp.json()


def save_all_companies(companies: dict, filename: str = "us_companies.json"):
    os.makedirs("stockInfo", exist_ok=True)
    path = os.path.join("stockInfo", filename)
    with open(path, "w") as f:
        json.dump(companies, f, indent=2)
    return path


# ====== Quarter extraction (durations) – quarter-only, keyed by end date ======

def extract_quarter_duration_by_enddate(gaap: dict, tags: list[str]) -> tuple[dict, dict]:
    """
    Extract quarter-only duration series from USD facts, keyed by end date 'YYYY-MM-DD'.

    - Accept forms: 10-Q, 10-K, 20-F, 40-F.
    - Accept fp in ("Q1","Q2","Q3","Q4","FY"), but:
        * Only keep contexts with duration between 75 and 110 days (approx one quarter).
          This picks true quarter-only values and ignores full-year FY (~365 days)
          and 6/9-month YTD contexts.
    - For each end date, keep the fact with the latest end/filed.

    Returns:
      quarterly: { 'YYYY-MM-DD' -> value }
      meta:      { 'YYYY-MM-DD' -> {... meta ...} }
    """
    quarterly: dict[str, float] = {}
    meta: dict[str, dict] = {}

    for tag in tags:
        tag_obj = gaap.get(tag)
        if not tag_obj:
            continue

        units = tag_obj.get("units", {})
        usd_items = units.get("USD")
        if not usd_items:
            continue

        for fact in usd_items:
            form = fact.get("form", "")
            if form not in ("10-Q", "10-K", "20-F", "40-F"):
                continue

            fp_raw = fact.get("fp")
            if not fp_raw:
                continue
            fp = str(fp_raw).upper()
            if fp not in ("Q1", "Q2", "Q3", "Q4", "FY"):
                continue

            start_str = fact.get("start")
            end_str = fact.get("end")
            if not start_str or not end_str:
                continue

            try:
                start_dt = datetime.fromisoformat(start_str[:10])
                end_dt = datetime.fromisoformat(end_str[:10])
            except Exception:
                continue

            duration_days = (end_dt - start_dt).days

            # Keep only ~1 quarter durations for quarterly series
            if duration_days < 75 or duration_days > 110:
                continue

            val = fact.get("val")
            if val is None:
                continue

            filed = fact.get("filed", "")
            accn = fact.get("accn", "")

            end_key = end_dt.date().isoformat()  # 'YYYY-MM-DD'

            if end_key not in quarterly:
                quarterly[end_key] = val
                meta[end_key] = {
                    "filed": filed,
                    "tag": tag,
                    "accn": accn,
                    "end": end_str,
                    "form": form,
                    "fp": fp,
                    "duration_days": duration_days,
                }
            else:
                existing = meta[end_key]
                best_end = existing.get("end", "")
                best_filed = existing.get("filed", "")
                # Prefer later end date; if tie, later filed date
                if (end_str and end_str > best_end) or (end_str == best_end and filed > best_filed):
                    quarterly[end_key] = val
                    meta[end_key] = {
                        "filed": filed,
                        "tag": tag,
                        "accn": accn,
                        "end": end_str,
                        "form": form,
                        "fp": fp,
                        "duration_days": duration_days,
                    }

    return quarterly, meta


# ====== Quarter extraction (instants) – assets/equity/debt by end date ======

def extract_quarter_instant_by_enddate(
    gaap: dict,
    tags: list[str],
) -> tuple[dict, dict]:
    """
    Extract quarter-end instant series (assets, equity, debt) from USD facts, keyed by end date.

    - Accept forms: 10-Q, 10-K, 20-F, 40-F.
    - Accept fp in ("Q1","Q2","Q3","Q4","FY").
    - Label by end date 'YYYY-MM-DD'.
    - If multiple instants for the same end date, prefer later end, then later filed.
    """
    quarterly: dict[str, float] = {}
    meta: dict[str, dict] = {}

    for tag in tags:
        tag_obj = gaap.get(tag)
        if not tag_obj:
            continue

        units = tag_obj.get("units", {})
        usd_items = units.get("USD")
        if not usd_items:
            continue

        for fact in usd_items:
            form = fact.get("form", "")
            if form not in ("10-Q", "10-K", "20-F", "40-F"):
                continue

            fp_raw = fact.get("fp")
            if not fp_raw:
                continue
            fp = str(fp_raw).upper()
            if fp not in ("Q1", "Q2", "Q3", "Q4", "FY"):
                continue

            end_str = fact.get("end")
            if not end_str:
                continue

            try:
                end_dt = datetime.fromisoformat(end_str[:10])
            except Exception:
                continue

            val = fact.get("val")
            if val is None:
                continue

            filed = fact.get("filed", "")
            accn = fact.get("accn", "")

            end_key = end_dt.date().isoformat()

            if end_key not in quarterly:
                quarterly[end_key] = val
                meta[end_key] = {
                    "filed": filed,
                    "tag": tag,
                    "accn": accn,
                    "end": end_str,
                    "form": form,
                    "fp": fp,
                }
            else:
                existing = meta[end_key]
                best_end = existing.get("end", "")
                best_filed = existing.get("filed", "")
                if (end_str and end_str > best_end) or (end_str == best_end and filed > best_filed):
                    quarterly[end_key] = val
                    meta[end_key] = {
                        "filed": filed,
                        "tag": tag,
                        "accn": accn,
                        "end": end_str,
                        "form": form,
                        "fp": fp,
                    }

    return quarterly, meta


# ====== Combine into quarterly metrics, keyed by end date ======

def extract_quarterly_metrics(gaap: dict) -> tuple[dict, dict, dict]:
    """
    Returns:
      metrics_by_period: { 'YYYY-MM-DD' -> {...metrics...} }
      revenue:           { 'YYYY-MM-DD' -> revenue_value }
      period_end_dates:  { 'YYYY-MM-DD' -> 'YYYY-MM-DD' end date (for convenience) }
    """
    revenue, revenue_meta = extract_quarter_duration_by_enddate(gaap, revenueTags)
    net_income, _ = extract_quarter_duration_by_enddate(gaap, netIncomeTags)
    op_cf, _ = extract_quarter_duration_by_enddate(gaap, operatingCashFlowTags)

    assets, _ = extract_quarter_instant_by_enddate(gaap, assetsTags)
    equity, _ = extract_quarter_instant_by_enddate(gaap, equityTags)
    debt, _ = extract_quarter_instant_by_enddate(gaap, debtLongTags)

    all_end_dates = (
        set(revenue.keys())
        | set(net_income.keys())
        | set(assets.keys())
        | set(equity.keys())
        | set(debt.keys())
        | set(op_cf.keys())
    )

    metrics_by_period: dict[str, dict] = {}
    period_end_dates: dict[str, str] = {}

    for end_key in sorted(all_end_dates):
        metrics_by_period[end_key] = {
            "revenue": revenue.get(end_key),
            "net_income": net_income.get(end_key),
            "assets": assets.get(end_key),
            "equity": equity.get(end_key),
            "debt": debt.get(end_key),
            "operating_cash_flow": op_cf.get(end_key),
        }

        meta = revenue_meta.get(end_key)
        if meta:
            period_end_dates[end_key] = meta.get("end", end_key)[:10]
        else:
            # Fallback: key itself is the end date
            period_end_dates[end_key] = end_key

    return metrics_by_period, revenue, period_end_dates


def find_internal_quarterly_gaps(revenue: dict) -> list[str]:
    """
    revenue: { 'YYYY-MM-DD' -> value }

    For debug: for each year, report if we don't see exactly 4 revenue quarters.
    Returns a list of human-readable messages like:
      "2017: 3 revenue quarters (['2017-01-29', '2017-04-30', '2017-07-30'])"
    """
    if not revenue:
        return []

    by_year: dict[int, list[str]] = {}

    for end_str in revenue.keys():
        if len(end_str) < 4:
            continue
        try:
            year = int(end_str[:4])
        except Exception:
            continue
        by_year.setdefault(year, []).append(end_str)

    messages: list[str] = []
    for y in sorted(by_year.keys()):
        dates = sorted(by_year[y])
        if len(dates) != 4:
            messages.append(f"{y}: {len(dates)} revenue quarters ({dates})")

    return messages


# ====== Helper to filter to 2012+ based on end date ======

def filter_2012_plus_quarters(d: dict) -> dict:
    """
    d: { 'YYYY-MM-DD' -> value }
    Keep only entries where year >= 2012.
    """
    out = {}
    for key, val in d.items():
        if len(key) < 4:
            continue
        try:
            year = int(key[:4])
        except Exception:
            continue
        if year >= 2012:
            out[key] = val
    return out


# ====== Group quarters into per-year structure (end-date keys) ======

def group_quarters_by_year(metrics_by_period: dict) -> dict:
    """
    Convert {'YYYY-MM-DD': metrics} into:
      {
        'YYYY': {
          'quarterly': {
             'YYYY-MM-DD': {...},   # quarter-end date within that year
             ...
          },
          'yearly': {
             revenue, net_income, assets, equity, debt, operating_cash_flow,
             stock_price (filled later)
          }
        },
        ...
      }

    - Year is determined from the quarter END DATE year.
    - Quarterly keys are the actual end date strings "YYYY-MM-DD".
    - Yearly aggregation:
        * revenue, net_income, operating_cash_flow = sum of quarterly values
        * assets, equity, debt = last non-null quarter in that year
    """
    years: dict[str, dict] = {}

    # 1) Group quarters under each calendar year by end date
    for end_key, metrics in metrics_by_period.items():
        # end_key is 'YYYY-MM-DD'
        if len(end_key) < 10:
            continue
        end_str = end_key[:10]
        try:
            year_int = int(end_str[:4])
        except Exception:
            continue
        year_key = str(year_int)

        if year_key not in years:
            years[year_key] = {
                "quarterly": {},
                "yearly": {},
            }

        years[year_key]["quarterly"][end_str] = metrics

    # 2) Build yearly aggregates (fundamentals only; stock_price added later)
    for year_key, yblock in years.items():
        quarters = sorted(
            yblock["quarterly"].items(),
            key=lambda kv: kv[0]  # sort by date string 'YYYY-MM-DD'
        )

        rev_vals = []
        ni_vals = []
        ocf_vals = []

        assets_last = None
        equity_last = None
        debt_last = None

        for end_date_str, m in quarters:
            rev = m.get("revenue")
            if rev is not None:
                rev_vals.append(rev)

            ni = m.get("net_income")
            if ni is not None:
                ni_vals.append(ni)

            ocf = m.get("operating_cash_flow")
            if ocf is not None:
                ocf_vals.append(ocf)

            if m.get("assets") is not None:
                assets_last = m["assets"]
            if m.get("equity") is not None:
                equity_last = m["equity"]
            if m.get("debt") is not None:
                debt_last = m["debt"]

        yearly_revenue = sum(rev_vals) if rev_vals else None
        yearly_net_income = sum(ni_vals) if ni_vals else None
        yearly_ocf = sum(ocf_vals) if ocf_vals else None

        yblock["yearly"] = {
            "revenue": yearly_revenue,
            "net_income": yearly_net_income,
            "assets": assets_last,
            "equity": equity_last,
            "debt": debt_last,
            "operating_cash_flow": yearly_ocf,
            "stock_price": None,  # filled by yfinance
        }

    return years


# ====== yfinance helpers – yearly only ======

def fetch_yearly_stock_prices_for_ticker(ticker: str, years: list[int]) -> dict:
    """
    Fetch stock price stats for ALL requested years with a SINGLE yfinance call.

    years: list of ints (e.g., [2012, 2013, 2014])

    Returns:
      { 'YYYY' -> {max_price, min_price, mean_price, median_price} }
    """
    if not years:
        return {}

    years = sorted(set(years))
    start_year = years[0]
    end_year = years[-1]

    start = f"{start_year}-01-01"
    end = f"{end_year + 1}-01-01"  # include full last year

    try:
        hist = yf.download(
            ticker,
            start=start,
            end=end,
            progress=False,
            auto_adjust=False,
        )
    except Exception as e:
        print(f"  ! yfinance failed for {ticker} between {start} and {end}: {e}")
        return {}

    if hist is None or hist.empty:
        print(f"  ! No price data for {ticker} between {start} and {end}")
        return {}

    # pick price column
    if "Adj Close" in hist.columns:
        price_series = hist["Adj Close"]
    elif "Close" in hist.columns:
        price_series = hist["Close"]
    else:
        print(f"  ! No Adj Close or Close column for {ticker} in {start}..{end}")
        return {}

    # Handle possible multi-column structure
    if hasattr(price_series, "columns"):
        price_series = price_series.iloc[:, 0]

    price_series = price_series.dropna().astype(float)

    result: dict[str, dict] = {}

    for y in years:
        y_str = str(y)
        year_slice = price_series.loc[f"{y}-01-01": f"{y}-12-31"]
        if year_slice is None or year_slice.empty:
            continue

        vals = year_slice
        max_price = float(vals.max())
        min_price = float(vals.min())
        mean_price = float(vals.mean())
        median_price = float(vals.median())

        result[y_str] = {
            "max_price": max_price,
            "min_price": min_price,
            "mean_price": mean_price,
            "median_price": median_price,
        }

    return result


# ====== Main script ======

if __name__ == "__main__":
    processing_limit = 10
    tickers = loadTickers()
    ticker_keys = list(tickers.keys())
    total = min(processing_limit, len(ticker_keys))
    bar = Bar("Processing", max=total)
    all_companies = {}

    rejected_companies = []

    for idx, key in enumerate(ticker_keys[:total]):
        cik_int = tickers[key]["cik_str"]
        ticker = tickers[key]["ticker"]
        company_name = tickers[key].get("title")
        cik = addTrailingZeros(cik_int)

        tickData = retrieveCompanyData(cik)
        if not tickData:
            reason = "companyfacts API call failed for this CIK"
            print(f"{ticker}: skipped ({reason})")
            rejected_companies.append({
                "ticker": ticker,
                "cik": cik,
                "name": company_name,
                "reason": reason,
            })
            bar.next()
            time.sleep(0.2)
            continue

        gaap = tickData.get("facts", {}).get("us-gaap")
        if not gaap:
            reason = "no us-gaap facts found in companyfacts"
            print(f"{ticker}: skipped ({reason})")
            rejected_companies.append({
                "ticker": ticker,
                "cik": cik,
                "name": company_name,
                "reason": reason,
            })
            bar.next()
            time.sleep(0.2)
            continue

        metrics_by_period, revenue_series, period_end_dates = extract_quarterly_metrics(gaap)

        # ---- Only care about 2012 and up ----
        metrics_by_period = filter_2012_plus_quarters(metrics_by_period)
        revenue_series = filter_2012_plus_quarters(revenue_series)
        period_end_dates = filter_2012_plus_quarters(period_end_dates)

        # 1) If there is no fiscal quarterly revenue from 2012 onward, skip this company.
        if not revenue_series:
            reason = "no fiscal quarterly revenue from 2012 onward found"
            print(f"{ticker}: skipped ({reason})")
            rejected_companies.append({
                "ticker": ticker,
                "cik": cik,
                "name": company_name,
                "reason": reason,
            })
            bar.next()
            time.sleep(0.2)
            continue

        # 2) Compute and PRINT revenue gaps (by year), but DO NOT reject for gaps.
        gaps = find_internal_quarterly_gaps(revenue_series)
        if gaps:
            print(f"{ticker} ({cik} – {company_name}): gaps in fiscal quarterly revenue: {gaps}")

        print(f"=== {ticker} ({cik}) ===")

        # Group quarters into per-year blocks using END DATE as quarterly key
        years_data = group_quarters_by_year(metrics_by_period)

        # Fetch YEARLY stock price stats with ONE yfinance call per ticker
        year_ints = [int(y) for y in years_data.keys()]
        yearly_prices = fetch_yearly_stock_prices_for_ticker(ticker, year_ints)

        # Attach stock_price only at the yearly level
        for year_str, yblock in years_data.items():
            sp = yearly_prices.get(year_str)
            yblock["yearly"]["stock_price"] = sp

        company_entry = {
            "cik": cik,
            "name": company_name,
        }
        company_entry.update(years_data)

        all_companies[ticker] = company_entry

        time.sleep(0.2)  # throttle SEC calls a bit
        bar.next()

    bar.finish()
    saved_path = save_all_companies(all_companies)
    print(f"Saved {len(all_companies)} companies to {saved_path}")

    print("\nRejected companies:")
    if not rejected_companies:
        print("None")
    else:
        for r in rejected_companies:
            print(f"{r['ticker']} ({r['cik']} – {r['name']}): {r['reason']}")

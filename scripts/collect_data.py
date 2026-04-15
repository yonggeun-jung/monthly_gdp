"""
collect_data.py
---------------
Collect macroeconomic data from FRED API and Yahoo Finance.

Usage:
    python scripts/collect_data.py
    python scripts/collect_data.py --country china
"""

import argparse
import os
import sys
import pandas as pd
from datetime import datetime

# FRED API key
FRED_API_KEY = "GET_YOUR_API_KEY"
# GET your own API key from FRED: https://fred.stlouisfed.org/docs/api/api_key.html

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data", "raw")

# GDP series (quarterly, from FRED)
GDP_SERIES = {
    "United States": ("GDPC1", datetime(1992, 1, 1)),
    "United Kingdom": ("NGDPRSAXDCGBQ", datetime(1991, 1, 1)),
    "Germany": ("CLVMNACSCAB1GQDE", datetime(1991, 1, 1)),
    "China": ("CHNGDPNQDSMEI", datetime(1992, 1, 1)),
}

# Monthly macro series (from FRED)
US_MONTHLY = {
    "working_hour_manuf": "AWHMAN",
    "CPI": "CPIAUCSL",
    "Ids_Prd": "INDPRO",
    "R_Csump": "DPCERA3M086SBEA",
    "Fed_Funds": "FEDFUNDS",
    "Intr_10Y": "GS10",
    "^GSPC_vol": "VIXCLS",
    "Unemp": "UNRATE",
    "All_Emp": "PAYEMS",
    "Labor_partic": "CIVPART",
    "M1": "M1SL",
    "M2": "WM2NS",
    "NetExports": "BOPGSTB",
    "Moody_aaa": "AAA",
}

UK_MONTHLY = {
    "Share_Prices": "SPASTT01GBM661N",
    "CPI": "GBRCPIALLMINMEI",
    "Prod_Vol": "GBRPROINDMISMEI",
    "Intr_Rate": "INTGSBGBM193N",
    "Intr_10Y": "IRLTLT01GBM156N",
    "Total Reserves": "TRESEGGBM052N",
    "Exchange": "EXUSUK",              
    "Imports": "XTIMVA01GBM664S",
    "Exports": "XTEXVA01GBM664S",
    "Unemp": "LRHUTTTTGBM156S",
}

CHINA_MONTHLY = {
    "CPI": "CPALTT01CNM657N",
    "PPI": "WPU1261",
    "Total_reserves": "TRESEGCNM052N",
    "Exports": "XTEXVA01CNM664S",
    "Imports": "XTIMVA01CNM664S",
    "Exchange": "CCUSMA02CNM618N",
    "Uncertainty": "CHNMAINLANDEPU",         
    "US imports": "XTIMVA01USM664S",
    "Eff. Exchange Rate": "RBCNBIS",       
}

GERMANY_MONTHLY = {
    "CPI": "DEUCPIALLMINMEI",
    "Unemp": "LRHUTTTTDEM156S",
    "Prod_Vol": "DEUPROINDMISMEI",
    "Retail": "DEUSARTMISMEI",
    "Exports": "XTEXVA01DEM664S",
    "Imports": "XTIMVA01DEM664S",
    "Total Reserves": "TRESEGDEM052N",
    "Exchange": "CCUSMA02DEM618N",
    "Eff_Ex_Rate": "RNDEBIS",              
    "Uncertainty": "EUEPUINDXM",            
}

# Stock index tickers (from Yahoo Finance)
STOCK_INDICES = {
    "United States": ("^GSPC", "S&P500", "1992-01-01"),
    "United Kingdom": ("^FTSE", "FTSE100", "1991-01-01"),
    "Germany": ("^GDAXI", "DAX", "1991-01-01"),
    # China (Shanghai Composite): downloaded manually from Investing.com
    # as Shanghai_Composite.csv -> place in data/raw/
}

COUNTRY_MONTHLY = {
    "United States": US_MONTHLY,
    "United Kingdom": UK_MONTHLY,
    "China": CHINA_MONTHLY,
    "Germany": GERMANY_MONTHLY,
}


def collect_gdp(api_key, country=None):
    """Collect quarterly GDP from FRED."""
    from fredapi import Fred
    fred = Fred(api_key=api_key)

    targets = {country: GDP_SERIES[country]} if country else GDP_SERIES
    for name, (series_id, start) in targets.items():
        print(f"  GDP: {name} ({series_id})")
        data = fred.get_series(series_id, observation_start=start)
        df = data.reset_index()
        df.columns = ["DATE", "Real GDP"]
        fname = f"quarterly_rgdp_{name.replace(' ', '_')}.csv"
        df.to_csv(os.path.join(OUTPUT_DIR, fname), index=False)


def collect_monthly_macro(api_key, country=None):
    """Collect monthly macro indicators from FRED."""
    from fredapi import Fred
    fred = Fred(api_key=api_key)

    targets = {country: COUNTRY_MONTHLY[country]} if country else COUNTRY_MONTHLY
    for name, series_dict in targets.items():
        print(f"  Monthly macro: {name}")
        frames = {}
        for var_name, series_id in series_dict.items():
            try:
                s = fred.get_series(series_id)
                frames[var_name] = s
            except Exception as e:
                print(f"    Warning: {var_name} ({series_id}): {e}")

        df = pd.DataFrame(frames)
        df.index.name = "DATE"
        slug = name.lower().replace(" ", "_")
        df.to_csv(os.path.join(OUTPUT_DIR, f"{slug}_monthly_macro_data.csv"))


def collect_stock_indices(country=None):
    """Collect stock index monthly averages from Yahoo Finance."""
    import yfinance as yf

    targets = {country: STOCK_INDICES[country]} if country else STOCK_INDICES
    for name, (ticker, col_name, start) in targets.items():
        print(f"  Stock index: {name} ({ticker})")
        try:
            data = yf.download(ticker, start=start, end="2025-12-31",
                               interval="1d", progress=False)
            monthly = data["Close"].resample("MS").mean()
            df = monthly.reset_index()
            df.columns = ["Date", col_name]
            slug = col_name.lower()
            df.to_csv(os.path.join(OUTPUT_DIR, f"{slug}_monthly_avg.csv"), index=False)
        except Exception as e:
            print(f"    Warning: {ticker}: {e}")


def main():
    parser = argparse.ArgumentParser(description="Collect macroeconomic data")
    parser.add_argument("--api_key", type=str, default=None,
                        help="FRED API key (overrides hardcoded key)")
    parser.add_argument("--country", type=str, default=None,
                        help="Single country or None for all")
    args = parser.parse_args()

    api_key = args.api_key or FRED_API_KEY
    if api_key == "GET_YOUR_API_KEY":
        print("ERROR: FRED API key is not set.")
        print("  Update FRED_API_KEY at the top of scripts/collect_data.py")
        print("  Get one at: https://fred.stlouisfed.org/docs/api/api_key.html")
        sys.exit(1)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("[1/3] Collecting GDP data...")
    collect_gdp(api_key, args.country)

    print("[2/3] Collecting monthly macro data...")
    collect_monthly_macro(api_key, args.country)

    print("[3/3] Collecting stock indices...")
    collect_stock_indices(args.country)

    print("Done. Files saved to", OUTPUT_DIR)


if __name__ == "__main__":
    main()
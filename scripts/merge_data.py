"""
merge_data.py
-------------
Merge collected data into master_<Country>.csv files,
generate summary statistics, ADF tests, and LaTeX tables.

Usage:
    python scripts/merge_data.py
    python scripts/merge_data.py --country china
"""

import argparse
import os
import sys
import pandas as pd
import numpy as np
import yaml

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
TAB_DIR = os.path.join(PROJECT_ROOT, "tab_tex")

MASTER_FILES = {
    "China": "master_China.csv",
    "Germany": "master_Germany.csv",
    "UK": "master_UK.csv",
    "US": "master_US.csv",
}

COUNTRY_CONFIG = {
    "China": {"file": "master_China.csv", "key": "china"},
    "Germany": {"file": "master_Germany.csv", "key": "germany"},
    "UK": {"file": "master_UK.csv", "key": "united_kingdom"},
    "US": {"file": "master_US.csv", "key": "united_states"},
}

VAR_INFO = {
    ("China", "CPI"): ("CPI (All Items, Growth)", "CPALTT01CNM657N"),
    ("China", "PPI"): ("PPI (Furniture, Household)", "WPU1261"),
    ("China", "Total_reserves"): ("Total Reserves (excl.\\ gold)", "TRESEGCNM052N"),
    ("China", "Exports"): ("Exports", "XTEXVA01CNM664S"),
    ("China", "Imports"): ("Imports", "XTIMVA01CNM664S"),
    ("China", "Exchange"): ("Exchange Rate (RMB/USD)", "EXCHUS"),
    ("China", "Uncertainty"): ("Policy Uncertainty", "CHNMAINLANDEPU"),
    ("China", "US imports"): ("US Imports from China", "IMPCH"),
    ("China", "Eff. Exchange Rate"): ("Effective Exchange Rate", "RBCNBIS"),
    ("China", "SSEC"): ("SSEC Index", "Investing.com"),
    ("China", "Real GDP"): ("Real GDP", "CHNGDPNQDSMEI"),
    ("Germany", "CPI"): ("CPI (All Items)", "DEUCPIALLMINMEI"),
    ("Germany", "Unemp"): ("Unemployment Rate", "LRHUTTTTDEM156S"),
    ("Germany", "Prod_Vol"): ("Production Volume", "DEUPROINDMISMEI"),
    ("Germany", "Retail"): ("Retail Trade", "DEUSARTMISMEI"),
    ("Germany", "Exports"): ("Exports", "XTEXVA01DEM664S"),
    ("Germany", "Imports"): ("Imports", "XTIMVA01DEM664S"),
    ("Germany", "Total Reserves"): ("Total Reserves (excl.\\ gold)", "TRESEGDEM052N"),
    ("Germany", "Exchange"): ("Exchange Rate", "CCUSMA02DEM650N"),
    ("Germany", "Eff_Ex_Rate"): ("Effective Exchange Rate", "RNDEBIS"),
    ("Germany", "Uncertainty"): ("Policy Uncertainty", "EUEPUINDXM"),
    ("Germany", "DAX"): ("DAX Index", "Yahoo Finance"),
    ("Germany", "Real GDP"): ("Real GDP", "CLVMNACSCAB1GQDE"),
    ("Germany", "price_comp"): ("Price Competitiveness", "Bundesbank"),
    ("UK", "Share_Prices"): ("Share Prices", "SPASTT01GBM661N"),
    ("UK", "CPI"): ("CPI (All Items)", "GBRCPIALLMINMEI"),
    ("UK", "Prod_Vol"): ("Production Volume", "GBRPROINDMISMEI"),
    ("UK", "Intr_Rate"): ("Interest Rate (Gov Bond)", "INTGSBGBM193N"),
    ("UK", "Intr_10Y"): ("Interest Rate (10Y Gov)", "IRLTLT01GBM156N"),
    ("UK", "Total Reserves"): ("Total Reserves (excl.\\ gold)", "TRESEGGBM052N"),
    ("UK", "Exchange"): ("Exchange Rate (GBP/USD)", "EXUSUK"),
    ("UK", "Imports"): ("Imports", "XTIMVA01GBM664S"),
    ("UK", "Exports"): ("Exports", "XTEXVA01GBM664S"),
    ("UK", "Unemp"): ("Unemployment Rate", "ONS"),
    ("UK", "FTSE100"): ("FTSE 100 Index", "Yahoo Finance"),
    ("UK", "Real GDP"): ("Real GDP", "NGDPRSAXDCGBQ"),
    ("UK", "M1"): ("M1 Money Stock", "BOE"),
    ("UK", "Emp"): ("Employment Rate", "ONS"),
    ("US", "working_hour_manuf"): ("Avg.\\ Weekly Hours, Manuf.", "AWHMAN"),
    ("US", "CPI"): ("CPI (All Items)", "CPIAUCSL"),
    ("US", "Ids_Prd"): ("Industrial Production", "INDPRO"),
    ("US", "R_Csump"): ("Real Personal Consum.", "DPCERA3M086SBEA"),
    ("US", "Fed_Funds"): ("Effective Fed Funds Rate", "FEDFUNDS"),
    ("US", "Intr_10Y"): ("Interest Rate (10Y Trea.)", "GS10"),
    ("US", "^GSPC_vol"): ("S\\&P 500 Volatility", "VIXCLS"),
    ("US", "Unemp"): ("Unemployment Rate", "UNRATE"),
    ("US", "All_Emp"): ("Nonfarm Payroll Emp.", "PAYEMS"),
    ("US", "Labor_partic"): ("Labor Force Participation", "CIVPART"),
    ("US", "M1"): ("M1 Money Stock", "M1SL"),
    ("US", "M2"): ("M2 Money Stock", "M2SL"),
    ("US", "NetExports"): ("Net Exports", "BOPTEXP"),
    ("US", "Moody_aaa"): ("Moody's AAA Bond Yield", "AAA"),
    ("US", "^GSPC"): ("S\\&P 500 Index", "Yahoo Finance"),
    ("US", "Real GDP"): ("Real GDP", "GDPC1"),
}


# Merge functions

def merge_us():
    print("  Merging US data...")
    gdp = pd.read_csv(os.path.join(RAW_DIR, "quarterly_rgdp_United_States.csv"),
                       parse_dates=["DATE"])
    macro = pd.read_csv(os.path.join(RAW_DIR, "united_states_monthly_macro_data.csv"),
                         parse_dates=["DATE"])
    # Resample daily series (e.g., VIXCLS) to monthly average
    macro["DATE"] = pd.to_datetime(macro["DATE"])
    macro = macro.set_index("DATE").resample("MS").mean().reset_index()
    sp500 = pd.read_csv(os.path.join(RAW_DIR, "s&p500_monthly_avg.csv"),
                         parse_dates=["Date"])

    sp500 = sp500.rename(columns={"Date": "DATE", "S&P500": "^GSPC"})
    sp500["DATE"] = sp500["DATE"].dt.to_period("M").dt.to_timestamp()
    gdp["DATE"] = gdp["DATE"].dt.to_period("M").dt.to_timestamp()

    merged = pd.merge(macro, sp500, on="DATE", how="outer")
    merged = pd.merge(merged, gdp, on="DATE", how="left")
    merged = merged.sort_values("DATE").reset_index(drop=True)
    merged.to_csv(os.path.join(DATA_DIR, "master_US.csv"), index=False)
    print(f"    Saved master_US.csv ({merged.shape})")


def merge_uk():
    print("  Merging UK data...")
    gdp = pd.read_csv(os.path.join(RAW_DIR, "quarterly_rgdp_United_Kingdom.csv"),
                       parse_dates=["DATE"])
    macro = pd.read_csv(os.path.join(RAW_DIR, "united_kingdom_monthly_macro_data.csv"),
                         parse_dates=["DATE"])
    ftse = pd.read_csv(os.path.join(RAW_DIR, "ftse100_monthly_avg.csv"),
                        parse_dates=["Date"])

    ftse = ftse.rename(columns={"Date": "DATE"})
    ftse["DATE"] = ftse["DATE"].dt.to_period("M").dt.to_timestamp()
    gdp["DATE"] = gdp["DATE"].dt.to_period("M").dt.to_timestamp()

    merged = pd.merge(macro, ftse, on="DATE", how="outer")
    merged = pd.merge(merged, gdp, on="DATE", how="left")

    for extra in ["uk_m1_emp.csv"]:
        path = os.path.join(RAW_DIR, extra)
        if os.path.exists(path):
            extra_df = pd.read_csv(path, parse_dates=["DATE"])
            merged = pd.merge(merged, extra_df, on="DATE", how="left")
            if "Unemp_y" in merged.columns:
                merged = merged.drop(columns=["Unemp_y"])
            if "Unemp_x" in merged.columns:
                merged = merged.rename(columns={"Unemp_x": "Unemp"})

    merged = merged.sort_values("DATE").reset_index(drop=True)
    merged.to_csv(os.path.join(DATA_DIR, "master_UK.csv"), index=False)
    print(f"    Saved master_UK.csv ({merged.shape})")


def merge_germany():
    print("  Merging Germany data...")
    gdp = pd.read_csv(os.path.join(RAW_DIR, "quarterly_rgdp_Germany.csv"),
                       parse_dates=["DATE"])
    macro = pd.read_csv(os.path.join(RAW_DIR, "germany_monthly_macro_data.csv"),
                         parse_dates=["DATE"])
    dax = pd.read_csv(os.path.join(RAW_DIR, "dax_monthly_avg.csv"),
                       parse_dates=["Date"])

    dax = dax.rename(columns={"Date": "DATE"})
    dax["DATE"] = dax["DATE"].dt.to_period("M").dt.to_timestamp()
    gdp["DATE"] = gdp["DATE"].dt.to_period("M").dt.to_timestamp()

    merged = pd.merge(macro, dax, on="DATE", how="outer")
    merged = pd.merge(merged, gdp, on="DATE", how="left")

    for extra in ["de_price_compet.csv"]:
        path = os.path.join(RAW_DIR, extra)
        if os.path.exists(path):
            extra_df = pd.read_csv(path, parse_dates=["DATE"])
            merged = pd.merge(merged, extra_df, on="DATE", how="left")

    merged = merged.sort_values("DATE").reset_index(drop=True)
    merged.to_csv(os.path.join(DATA_DIR, "master_Germany.csv"), index=False)
    print(f"    Saved master_Germany.csv ({merged.shape})")


def merge_china():
    print("  Merging China data...")
    gdp = pd.read_csv(os.path.join(RAW_DIR, "quarterly_rgdp_China.csv"),
                       parse_dates=["DATE"])
    macro = pd.read_csv(os.path.join(RAW_DIR, "china_monthly_macro_data.csv"),
                         parse_dates=["DATE"])

    gdp["DATE"] = gdp["DATE"].dt.to_period("M").dt.to_timestamp()

    ssec_path = os.path.join(RAW_DIR, "Shanghai_Composite.csv")
    if os.path.exists(ssec_path):
        ssec = pd.read_csv(ssec_path, parse_dates=["Date"])
        ssec = ssec.rename(columns={"Date": "DATE", "ssec": "SSEC"})
        ssec["DATE"] = ssec["DATE"].dt.to_period("M").dt.to_timestamp()
        merged = pd.merge(macro, ssec, on="DATE", how="outer")
    else:
        print("    Warning: Shanghai_Composite.csv not found, skipping SSEC")
        merged = macro.copy()

    merged = pd.merge(merged, gdp, on="DATE", how="left")
    merged = merged.sort_values("DATE").reset_index(drop=True)
    merged = merged[merged["DATE"] >= "1994-01-01"]
    merged.to_csv(os.path.join(DATA_DIR, "master_China.csv"), index=False)
    print(f"    Saved master_China.csv ({merged.shape})")


MERGE_FNS = {
    "united_states": merge_us,
    "united_kingdom": merge_uk,
    "germany": merge_germany,
    "china": merge_china,
}


# Summary statistics and ADF

def generate_summary_stats():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    config = load_yaml()
    all_rows = []
    for country, info in COUNTRY_CONFIG.items():
        path = os.path.join(DATA_DIR, info["file"])
        if not os.path.exists(path):
            print(f"    Warning: {info['file']} not found, skipping {country}")
            continue
        df = pd.read_csv(path)
        # Filter to analysis period
        cfg_key = info["key"]
        start = config.get(cfg_key, {}).get("start_date")
        if start:
            df["DATE"] = pd.to_datetime(df["DATE"])
            df = df[df["DATE"] >= start].reset_index(drop=True)
        exclude = {"DATE", "quarter"}
        numeric = [c for c in df.columns if c not in exclude
                   and pd.api.types.is_numeric_dtype(df[c])]
        gdp_col = [c for c in numeric if "GDP" in c]
        monthly_cols = [c for c in numeric if c not in gdp_col]
        df_aligned = df.dropna(subset=monthly_cols).reset_index(drop=True)
        for col in numeric:
            if col in gdp_col:
                series = pd.to_numeric(df[col], errors="coerce").dropna()
            else:
                series = pd.to_numeric(df_aligned[col], errors="coerce").dropna()
            if len(series) == 0:
                continue
            all_rows.append({
                "Country": country, "Variable": col,
                "Obs": int(len(series)), "Mean": series.mean(),
                "Std": series.std(), "Min": series.min(), "Max": series.max(),
            })
    stats_df = pd.DataFrame(all_rows)
    out_path = os.path.join(RESULTS_DIR, "data_summary_stats.csv")
    stats_df.to_csv(out_path, index=False)
    print(f"    Saved summary stats ({len(stats_df)} rows) to {out_path}")


def generate_adf_results():
    from statsmodels.tsa.stattools import adfuller
    os.makedirs(RESULTS_DIR, exist_ok=True)
    all_rows = []
    for country, fname in MASTER_FILES.items():
        path = os.path.join(DATA_DIR, fname)
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path)
        exclude = {"DATE", "quarter"}
        for col in df.columns:
            if col in exclude or not pd.api.types.is_numeric_dtype(df[col]):
                continue
            series = pd.to_numeric(df[col], errors="coerce").dropna()
            if len(series) < 20:
                continue
            try:
                result = adfuller(series)
                all_rows.append({
                    "Country": country, "Variable": col,
                    "ADF_Stat": round(result[0], 4),
                    "p_value": round(result[1], 4),
                    "Lags_Used": result[2], "Nobs": result[3],
                    "Stationary_5pct": result[1] < 0.05,
                })
            except Exception as e:
                print(f"    Warning: ADF failed for {country}/{col}: {e}")
    adf_df = pd.DataFrame(all_rows)
    out_path = os.path.join(RESULTS_DIR, "adf_results.csv")
    adf_df.to_csv(out_path, index=False)
    print(f"    Saved ADF results ({len(adf_df)} rows) to {out_path}")


# LaTeX table generation

def load_yaml():
    path = os.path.join(PROJECT_ROOT, "config", "countries.yaml")
    with open(path) as f:
        return yaml.safe_load(f)


def get_transform(col, country_key, config):
    cfg = config[country_key]
    log_diff = cfg.get("log_diff_cols", [])
    diff = cfg.get("diff_cols", [])
    target = cfg.get("target_col", "Real GDP")
    if col == target:
        return "---", "$\\Delta\\log(\\text{GDP}_q)$"
    elif col in log_diff:
        return "Yes", "$\\Delta\\log(X_m)$"
    elif col in diff:
        return "Yes", "$\\Delta X_m$"
    else:
        return "No", "$(X_m - \\mu_j)/\\sigma_j$"


def fmt_num(x):
    ax = abs(x)
    if ax == 0:
        return "0"
    elif ax >= 1e9:
        return f"{x:.2e}"
    elif ax >= 1000:
        return f"{x:,.0f}"
    elif ax >= 10:
        return f"{x:.2f}"
    elif ax >= 1:
        return f"{x:.3f}"
    elif ax >= 0.01:
        return f"{x:.4f}"
    else:
        return f"{x:.2e}"


def generate_dataset_table(config):
    os.makedirs(TAB_DIR, exist_ok=True)
    lines = []
    lines.append(r"\begin{longtable}{llcc}")
    lines.append(r"\caption{Dataset Description by Country} \label{tab:dataset_all} \\")
    lines.append(r"\toprule")
    lines.append(r"Variable & Code & Unit Root & Transform. \\")
    lines.append(r"\midrule")
    lines.append(r"\endfirsthead")
    lines.append(r"\toprule")
    lines.append(r"Variable & Code & Unit Root & Transform. \\")
    lines.append(r"\midrule")
    lines.append(r"\endhead")
    lines.append(r"\bottomrule")
    lines.append(r"\endfoot")

    for country, info in COUNTRY_CONFIG.items():
        path = os.path.join(DATA_DIR, info["file"])
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path, nrows=1)
        cols = [c for c in df.columns if c not in ("DATE", "quarter")]

        lines.append(f"\\textit{{{country}}} & & & \\\\")
        for col in cols:
            key = (country, col)
            display, code = VAR_INFO.get(key, (col, "---"))
            unit_root, transform = get_transform(col, info["key"], config)

            safe_sources = {"---", "Investing.com", "Yahoo Finance",
                            "Bundesbank", "BOE", "ONS"}
            code_esc = code if code in safe_sources else code.replace("_", "\\_")

            lines.append(
                f"{display} & \\texttt{{{code_esc}}} "
                f"& {unit_root} & {transform} \\\\"
            )
        lines.append(r"\addlinespace")

    lines.append(r"\end{longtable}")
    lines.append(r"\vspace{0.5em}")
    lines.append(r"\noindent{\footnotesize \textit{Note}: Most series are from FRED. "
                 r"Stock indices are from Yahoo Finance. "
                 r"SSEC Index is from Investing.com. "
                 r"UK M1 and employment data are from BOE/ONS. "
                 r"Germany price competitiveness is from the Deutsche Bundesbank. "
                 r"``Unit Root'' indicates non-stationarity based on the "
                 r"Augmented Dickey-Fuller test at the 5\% level and economic considerations. "
                 r"$\Delta\log(X_m)$: log-difference; "
                 r"$\Delta X_m$: first-difference (applied to interest rates); "
                 r"$(X_m - \mu_j)/\sigma_j$: Z-score standardization.}"
    )

    out = os.path.join(TAB_DIR, "dataset_all.tex")
    with open(out, "w") as f:
        f.write("\n".join(lines))
    print(f"    Saved {out}")


def generate_stats_table():
    os.makedirs(TAB_DIR, exist_ok=True)
    config = load_yaml()
    lines = []
    lines.append(r"\begin{longtable}{lrrrrr}")
    lines.append(r"\caption{Summary Statistics by Country} \label{tab:data_stat_all} \\")
    lines.append(r"\toprule")
    lines.append(r"Variable & Obs. & Mean & Std.\ Dev. & Min & Max \\")
    lines.append(r"\midrule")
    lines.append(r"\endfirsthead")
    lines.append(r"\toprule")
    lines.append(r"Variable & Obs. & Mean & Std.\ Dev. & Min & Max \\")
    lines.append(r"\midrule")
    lines.append(r"\endhead")
    lines.append(r"\bottomrule")
    lines.append(r"\endfoot")

    for country, info in COUNTRY_CONFIG.items():
        path = os.path.join(DATA_DIR, info["file"])
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path)
        # Filter to analysis period
        cfg_key = info["key"]
        start = config.get(cfg_key, {}).get("start_date")
        if start:
            df["DATE"] = pd.to_datetime(df["DATE"])
            df = df[df["DATE"] >= start].reset_index(drop=True)
        cols = [c for c in df.columns if c not in ("DATE", "quarter")]
        gdp_col = [c for c in cols if "GDP" in c]
        monthly_cols = [c for c in cols if c not in gdp_col]
        df_aligned = df.dropna(subset=monthly_cols).reset_index(drop=True)

        lines.append(f"\\textit{{{country}}} & & & & & \\\\")
        for col in cols:
            key = (country, col)
            display, _ = VAR_INFO.get(key, (col, ""))
            if col in gdp_col:
                series = pd.to_numeric(df[col], errors="coerce").dropna()
            else:
                series = pd.to_numeric(df_aligned[col], errors="coerce").dropna()
            if len(series) == 0:
                continue

            lines.append(
                f"{display} & {len(series)} "
                f"& {fmt_num(series.mean())} & {fmt_num(series.std())} "
                f"& {fmt_num(series.min())} & {fmt_num(series.max())} \\\\"
            )
        lines.append(r"\addlinespace")

    lines.append(r"\end{longtable}")
    lines.append(r"\noindent{\footnotesize \textit{Note}: Summary statistics are calculated "
                 r"using raw values before any transformation "
                 r"(e.g., differencing or standardization).}"
    )

    out = os.path.join(TAB_DIR, "data_stat_all.tex")
    with open(out, "w") as f:
        f.write("\n".join(lines))
    print(f"    Saved {out}")


# Main

def main():
    parser = argparse.ArgumentParser(description="Merge data into master CSVs")
    parser.add_argument("--country", type=str, default=None)
    args = parser.parse_args()

    # Step 1: Merge
    print("[1/5] Merging data...")
    if args.country:
        MERGE_FNS[args.country]()
    else:
        for fn in MERGE_FNS.values():
            fn()

    # Step 2: Summary stats
    print("\n[2/5] Generating summary statistics...")
    generate_summary_stats()

    # Step 3: ADF tests
    print("[3/5] Running ADF tests...")
    generate_adf_results()

    # Step 4-5: LaTeX tables
    print("[4/5] Generating dataset description table...")
    config = load_yaml()
    generate_dataset_table(config)

    print("[5/5] Generating summary statistics table...")
    generate_stats_table()

    print("Done.")


if __name__ == "__main__":
    main()
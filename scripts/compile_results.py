"""
compile_results.py
------------------
Read all metrics/ew CSVs and produce:
  1. results/summary_all.csv          (all models x countries x lags)
  2. results/dm_tests_all.csv         (pairwise DM tests, all lags)

Usage:
    python scripts/compile_results.py
"""

import os
import re
import sys
import glob
import numpy as np
import pandas as pd
import itertools

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

COUNTRY_ORDER = ["china", "germany", "united_kingdom", "united_states"]
COUNTRY_LABELS = {
    "china": "China",
    "germany": "Germany",
    "united_kingdom": "UK",
    "united_states": "US",
}
MODEL_ORDER = ["chow_lin", "elastic_net", "xgboost", "mlp"]
MODEL_LABELS = {
    "chow_lin": "Chow-Lin",
    "elastic_net": "Elastic Net",
    "xgboost": "XGBoost",
    "mlp": "MLP",
}


def _parse_metrics_fname(basename: str):
    """
    metrics_CN_mlp_lag0.csv -> (model='mlp', lag=0)
    metrics_mlp_lag0.csv (legacy) -> (model='mlp', lag=0)
    """
    stem = basename.replace(".csv", "")
    if not stem.startswith("metrics_"):
        return None, None
    rest = stem[len("metrics_") :]
    m = re.match(r"^([A-Z]{2})_(.+)_lag(\d+)$", rest)
    if m:
        return m.group(2), int(m.group(3))
    m2 = re.match(r"^(.+)_lag(\d+)$", rest)
    if m2:
        return m2.group(1), int(m2.group(2))
    return None, None


def load_all_metrics():
    """Read all metrics_*_lag*.csv files into a single DataFrame."""
    rows = []
    for f in sorted(glob.glob(os.path.join(RESULTS_DIR, "*/metrics_*_lag*.csv"))):
        parts = f.replace(RESULTS_DIR + "/", "").split("/")
        country = parts[0]
        model, lag = _parse_metrics_fname(parts[1])
        if model is None:
            continue
        df = pd.read_csv(f)
        df["Country"] = country
        df["Model"] = model
        df["Lag"] = int(lag)
        rows.append(df)

    if not rows:
        print("No metrics files found.")
        sys.exit(1)

    return pd.concat(rows, ignore_index=True)


def load_ew_predictions(country, model, lag):
    """Load expanding window predictions for a country-model-lag combo."""
    cc_map = {
        "china": "CN",
        "germany": "DE",
        "united_kingdom": "UK",
        "united_states": "US",
    }
    cc = cc_map.get(country, country[:2].upper())
    path_new = os.path.join(RESULTS_DIR, country, f"ew_{cc}_{model}_lag{lag}.csv")
    path_old = os.path.join(RESULTS_DIR, country, f"ew_{model}_lag{lag}.csv")
    for path in (path_new, path_old):
        if os.path.exists(path):
            return pd.read_csv(path)
    return None


def run_dm_test(y_true, y_pred_1, y_pred_2, h=1):
    """Diebold-Mariano test with HAC variance."""
    from scipy import stats

    e1 = (y_true - y_pred_1) ** 2
    e2 = (y_true - y_pred_2) ** 2
    d = e1 - e2
    n = len(d)
    d_bar = np.mean(d)

    gamma_0 = np.mean((d - d_bar) ** 2)
    gamma_sum = 0.0
    for k in range(1, h):
        gamma_k = np.mean((d[k:] - d_bar) * (d[:-k] - d_bar))
        gamma_sum += 2.0 * gamma_k
    var_d = (gamma_0 + gamma_sum) / n

    if var_d <= 0:
        return np.nan, np.nan, None

    dm_stat = d_bar / np.sqrt(var_d)
    p_value = 2.0 * (1.0 - stats.norm.cdf(np.abs(dm_stat)))
    return dm_stat, p_value, d_bar < 0


def pairwise_dm_tests(country, models, lag):
    """Run all pairwise DM tests for a country at a given lag."""
    preds = {}
    for m in models:
        df = load_ew_predictions(country, m, lag)
        if df is not None:
            valid = df.dropna(subset=["Predicted"])
            if len(valid) > 0:
                preds[m] = valid

    results = []
    model_list = [m for m in MODEL_ORDER if m in preds]

    for m1, m2 in itertools.combinations(model_list, 2):
        df1 = preds[m1]
        df2 = preds[m2]
        common = df1.merge(df2, on="Quarter", suffixes=("_1", "_2"))
        if len(common) < 5:
            continue

        dm_stat, p_val, m1_better = run_dm_test(
            common["Actual_1"].values,
            common["Predicted_1"].values,
            common["Predicted_2"].values,
        )

        results.append({
            "Country": COUNTRY_LABELS.get(country, country),
            "Lag": lag,
            "Model_1": MODEL_LABELS.get(m1, m1),
            "Model_2": MODEL_LABELS.get(m2, m2),
            "DM_Stat": dm_stat,
            "p_value": p_val,
            "Favors": MODEL_LABELS.get(m1, m1) if m1_better else MODEL_LABELS.get(m2, m2),
            "Sig_5pct": p_val < 0.05 if not np.isnan(p_val) else False,
        })

    return pd.DataFrame(results)


def main():
    print("Loading all metrics...")
    all_df = load_all_metrics()

    # 1. Full summary
    cols = ["Country", "Model", "Lag", "RMSE", "MAE", "R2",
            "Correlation", "Sign Accuracy", "sMAPE", "Theil U1"]
    summary = all_df[[c for c in cols if c in all_df.columns]].copy()

    # Sort
    summary["c_order"] = summary["Country"].map(
        {c: i for i, c in enumerate(COUNTRY_ORDER)})
    summary["m_order"] = summary["Model"].map(
        {m: i for i, m in enumerate(MODEL_ORDER)})
    summary = summary.sort_values(["c_order", "Lag", "m_order"])
    summary = summary.drop(columns=["c_order", "m_order"])

    out1 = os.path.join(RESULTS_DIR, "summary_all.csv")
    summary.to_csv(out1, index=False)
    print(f"  Saved {out1}")

    # Print summary
    print("\n" + "=" * 90)
    print("  ALL RESULTS")
    print("=" * 90)
    display_cols = ["Country", "Model", "Lag", "RMSE", "MAE",
                    "R2", "Correlation", "Sign Accuracy"]
    print(summary[[c for c in display_cols
                    if c in summary.columns]].to_string(index=False))

    # 2. DM tests for all lags
    print("\n" + "=" * 90)
    print("  DIEBOLD-MARIANO TESTS")
    print("=" * 90)
    all_lags = sorted(summary["Lag"].unique())
    dm_rows = []
    for lag in all_lags:
        for country in COUNTRY_ORDER:
            dm = pairwise_dm_tests(country, MODEL_ORDER, lag)
            if len(dm) > 0:
                dm_rows.append(dm)
                for _, row in dm.iterrows():
                    sig = "*" if row["Sig_5pct"] else ""
                    print(
                        f"  Lag={row['Lag']}  {row['Country']:8s}  "
                        f"{row['Model_1']:12s} vs {row['Model_2']:12s}  "
                        f"DM={row['DM_Stat']:7.3f}  p={row['p_value']:.3f}{sig}  "
                        f"-> {row['Favors']}"
                    )

    if dm_rows:
        dm_all = pd.concat(dm_rows, ignore_index=True)
        out2 = os.path.join(RESULTS_DIR, "dm_tests_all.csv")
        dm_all.to_csv(out2, index=False)
        print(f"\n  Saved {out2}")

    # 3. Missing combinations
    print("\n" + "=" * 90)
    print("  MISSING RESULTS")
    print("=" * 90)
    existing = set(zip(summary["Country"], summary["Model"], summary["Lag"]))
    missing = []
    for country in COUNTRY_ORDER:
        for model in MODEL_ORDER:
            for lag in all_lags:
                if (country, model, lag) not in existing:
                    missing.append(f"  {country} | {model} | lag={lag}")
    if missing:
        for m in missing:
            print(m)
    else:
        print("  None - all combinations complete!")

    print("\nDone.")


if __name__ == "__main__":
    main()

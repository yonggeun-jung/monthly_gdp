"""
Figure Generation for Paper
----------------------------
Run from the project root directory.
Requires: matplotlib, pandas, numpy, seaborn

Usage:
    python generate_figures.py \
        --us_monthly  results/united_states/monthly_gdp_elastic_net_lag1.csv \
        --uk_monthly  results/united_kingdom/monthly_gdp_chow_lin_lag0.csv \
        --koop        data/raw/koop_montly_gdp.csv \
        --ons         data/raw/ons_montly_gdp.csv \
        --summary     results/summary_all.csv \
        --outdir      figures/
"""

import argparse
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.colors import LinearSegmentedColormap

# ── Style ──────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "Nimbus Sans", "DejaVu Sans"],
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "legend.fontsize": 10,
    "figure.dpi": 300,
    "pdf.fonttype": 42,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.1,
})
CRISIS_COLOR = "#e5e7eb"      # light gray for recession shading
COLORS = {
    "chow_lin": "#1d4ed8",    # blue
    "elastic_net": "#dc2626", # red
    "xgboost": "#4b5563",     # gray
    "mlp": "#111111",         # black
    "benchmark": "#6b7280",   # gray
    "actual_q": "#9ca3af",    # light gray
    "ci": "#9ca3af",          # confidence interval fill
}


def resolve_input_path(user_path, candidates, label):
    """Resolve an input path from CLI or fallback candidates."""
    if user_path:
        if os.path.exists(user_path):
            return user_path
        raise FileNotFoundError(f"{label} not found: {user_path}")
    for path in candidates:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(
        f"{label} not found. Provide --{label} explicitly."
    )


def resolve_optional_path(user_path, candidates):
    """Resolve optional input; return None if not found."""
    if user_path:
        return user_path if os.path.exists(user_path) else None
    for path in candidates:
        if os.path.exists(path):
            return path
    return None


def shade_crises(ax, date_min, date_max):
    """Add shaded rectangles for GFC and COVID periods."""
    crises = [
        (pd.Timestamp("2008-01-01"), pd.Timestamp("2009-06-30"), "GFC"),
        (pd.Timestamp("2020-01-01"), pd.Timestamp("2020-12-31"), "COVID-19"),
    ]
    for start, end, label in crises:
        if start < date_max and end > date_min:
            s = max(start, date_min)
            e = min(end, date_max)
            ax.axvspan(s, e, alpha=0.25, color=CRISIS_COLOR, zorder=0)


# ── Figure 1: US Monthly GDP Level ────────────────────────
def fig_us_monthly_level(us_path, outdir):
    """Monthly GDP level from Elastic Net lag=1."""
    df = pd.read_csv(us_path, parse_dates=["DATE"])
    df = df.dropna(subset=["monthly_level"])

    fig, ax = plt.subplots(figsize=(12, 4.5))
    shade_crises(ax, df["DATE"].min(), df["DATE"].max())
    ax.plot(df["DATE"], df["monthly_level"], color=COLORS["elastic_net"],
            linewidth=1.2, label="Elastic Net (lag 1)")
    ax.set_ylabel("Monthly GDP Level")
    ax.set_xlabel("")
    ax.xaxis.set_major_locator(mdates.YearLocator(5))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.legend(frameon=False)
    ax.set_title("United States: Estimated Monthly GDP Level")
    fig.savefig(f"{outdir}/fig_us_monthly_level.pdf")
    plt.close(fig)
    print("  Saved: fig_us_monthly_level")


# ── Figure 2: US Monthly Growth ───────────────────────────
def fig_us_monthly_growth(us_path, outdir):
    """Monthly GDP growth from Elastic Net lag=1."""
    df = pd.read_csv(us_path, parse_dates=["DATE"])
    df = df.dropna(subset=["adjusted_growth"])

    fig, ax = plt.subplots(figsize=(12, 4.5))
    shade_crises(ax, df["DATE"].min(), df["DATE"].max())
    ax.plot(df["DATE"], df["adjusted_growth"] * 100,
            color=COLORS["elastic_net"], linewidth=0.8,
            label="Monthly growth (Elastic Net, lag 1)")
    ax.axhline(0, color="black", linewidth=0.5, linestyle="-")
    ax.set_ylabel("Monthly Growth (%)")
    ax.set_xlabel("")
    ax.xaxis.set_major_locator(mdates.YearLocator(5))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.legend(frameon=False)
    ax.set_title("United States: Estimated Monthly GDP Growth")
    fig.savefig(f"{outdir}/fig_us_monthly_growth.pdf")
    plt.close(fig)
    print("  Saved: fig_us_monthly_growth")


# ── Figure 3: US Benchmark Comparison (Koop et al.) ──────
def fig_us_benchmark(us_path, koop_path, outdir):
    """
    Compare annualized monthly GDP growth with Koop et al. (2023).
    Koop uses decimal year dates and reports annualized growth.
    Our series: apply MA(5) weights × 4 × 100 for annualization.
    """
    # Load our estimates
    df = pd.read_csv(us_path, parse_dates=["DATE"])
    df = df.dropna(subset=["adjusted_growth"])
    y = df["adjusted_growth"].values

    # Annualize via MA(5): (1/3 y_t + 2/3 y_{t-1} + y_{t-2}
    #                        + 2/3 y_{t-3} + 1/3 y_{t-4}) × 4 × 100
    ann = np.full(len(y), np.nan)
    for t in range(4, len(y)):
        ann[t] = (y[t]/3 + 2*y[t-1]/3 + y[t-2]
                  + 2*y[t-3]/3 + y[t-4]/3) * 4 * 100
    df["annualized"] = ann

    # Load Koop
    koop = pd.read_csv(koop_path)
    # Convert decimal year to datetime
    koop_years = koop["Date"].values
    koop_dates = pd.to_datetime(
        [f"{int(y)}-{int(round((y % 1) * 12)) + 1:02d}-01"
         for y in koop_years]
    )
    koop["date"] = koop_dates

    # Overlap period
    overlap_start = max(df["DATE"].min(), koop["date"].min())
    overlap_end = min(df["DATE"].max(), koop["date"].max())

    ours = df[(df["DATE"] >= overlap_start) & (df["DATE"] <= overlap_end)]
    theirs = koop[(koop["date"] >= overlap_start) & (koop["date"] <= overlap_end)]

    fig, ax = plt.subplots(figsize=(12, 5))
    shade_crises(ax, overlap_start, overlap_end)

    ax.plot(ours["DATE"], ours["annualized"],
            color=COLORS["elastic_net"], linewidth=1.2,
            label="This paper (Elastic Net, lag 1)")
    ax.plot(theirs["date"], theirs["GDP"],
            color=COLORS["benchmark"], linewidth=1.2, linestyle="--",
            label="Koop et al. (2023)")
    ax.fill_between(theirs["date"],
                     theirs["16_percentile"], theirs["84_percentile"],
                     alpha=0.10, color=COLORS["ci"], zorder=1,
                     label="Koop et al. 68% CI")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_ylabel("Annualized Monthly GDP Growth (%)")
    ax.set_xlabel("")
    ax.xaxis.set_major_locator(mdates.YearLocator(5))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.legend(frameon=False, loc="lower left")
    ax.set_title("United States: Comparison with Koop et al. (2023)")

    # Correlation on overlap
    merged = pd.merge(
        ours[["DATE", "annualized"]].dropna(),
        theirs[["date", "GDP"]].rename(columns={"date": "DATE"}),
        on="DATE", how="inner"
    )
    if len(merged) > 5:
        corr = merged["annualized"].corr(merged["GDP"])
        ax.text(0.98, 0.95, f"Correlation: {corr:.3f}",
                transform=ax.transAxes, fontsize=10,
                horizontalalignment="right",
                verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

    fig.savefig(f"{outdir}/fig_us_benchmark_koop.pdf")
    plt.close(fig)
    print("  Saved: fig_us_benchmark_koop")


# ── Figure 4: UK Benchmark Comparison (ONS) ──────────────
def fig_uk_benchmark(uk_path, ons_path, outdir):
    """Compare UK monthly GDP level with ONS official series."""
    # Load our estimates
    df = pd.read_csv(uk_path, parse_dates=["DATE"])
    df = df.dropna(subset=["monthly_level"])

    # Load ONS
    ons_raw = pd.read_csv(ons_path)
    # Find the GVA index column and the date column
    gva_col = "Gross Value Added - Monthly (Index 1dp) :CVM SA"
    date_col = "Title"

    # Skip metadata rows (find first row that looks like a date)
    data_start = None
    for i, val in enumerate(ons_raw[date_col]):
        if isinstance(val, str) and len(val) == 8 and val[:4].isdigit():
            data_start = i
            break

    if data_start is None:
        print("  ERROR: Could not find data start in ONS file")
        return

    ons = ons_raw.iloc[data_start:].copy()
    ons["gva"] = pd.to_numeric(ons[gva_col], errors="coerce")

    # Parse ONS dates: "1997 JAN" format
    month_map = {"JAN": 1, "FEB": 2, "MAR": 3, "APR": 4, "MAY": 5,
                 "JUN": 6, "JUL": 7, "AUG": 8, "SEP": 9, "OCT": 10,
                 "NOV": 11, "DEC": 12}
    dates = []
    for d in ons[date_col]:
        parts = str(d).strip().split()
        if len(parts) == 2 and parts[1] in month_map:
            dates.append(pd.Timestamp(f"{parts[0]}-{month_map[parts[1]]:02d}-01"))
        else:
            dates.append(pd.NaT)
    ons["date"] = dates
    ons = ons.dropna(subset=["date", "gva"])

    # Rescale our series to match ONS base (Oct 2022 = 100)
    ons_base = ons.loc[ons["date"] == pd.Timestamp("2022-10-01"), "gva"]
    if len(ons_base) == 0:
        # Fallback: use closest date
        ons_base_val = 100.0
    else:
        ons_base_val = float(ons_base.iloc[0])

    our_base = df.loc[df["DATE"] == pd.Timestamp("2022-10-01"), "monthly_level"]
    if len(our_base) == 0:
        print("  WARNING: No Oct 2022 in our data, using last available")
        our_base_val = df["monthly_level"].iloc[-1]
    else:
        our_base_val = float(our_base.iloc[0])

    df["rescaled"] = df["monthly_level"] / our_base_val * ons_base_val

    # Overlap
    overlap_start = max(df["DATE"].min(), ons["date"].min())
    overlap_end = min(df["DATE"].max(), ons["date"].max())

    ours = df[(df["DATE"] >= overlap_start) & (df["DATE"] <= overlap_end)]
    theirs = ons[(ons["date"] >= overlap_start) & (ons["date"] <= overlap_end)]

    fig, ax = plt.subplots(figsize=(12, 5))
    shade_crises(ax, overlap_start, overlap_end)

    ax.plot(ours["DATE"], ours["rescaled"],
            color=COLORS["chow_lin"], linewidth=1.2,
            label="This paper (Chow-Lin, lag 0)")
    ax.plot(theirs["date"], theirs["gva"],
            color=COLORS["benchmark"], linewidth=1.2, linestyle="--",
            label="ONS Official Monthly GDP")
    ax.set_ylabel("Index (Oct 2022 = 100)")
    ax.set_xlabel("")
    ax.xaxis.set_major_locator(mdates.YearLocator(5))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.legend(frameon=False, loc="lower right")
    ax.set_title("United Kingdom: Comparison with ONS Monthly GDP")

    # Correlation
    merged = pd.merge(
        ours[["DATE", "rescaled"]].dropna(),
        theirs[["date", "gva"]].rename(columns={"date": "DATE"}),
        on="DATE", how="inner"
    )
    if len(merged) > 5:
        corr = merged["rescaled"].corr(merged["gva"])
        ax.text(0.02, 0.95, f"Correlation: {corr:.3f}",
                transform=ax.transAxes, fontsize=10,
                verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

    fig.savefig(f"{outdir}/fig_uk_benchmark_ons.pdf")
    plt.close(fig)
    print("  Saved: fig_uk_benchmark_ons")


# ── Figure 5: R2 comparison across countries and lags ─────
def fig_r2_heatmap(summary_path, outdir):
    """Heatmap of R2 across countries, models, and lags."""
    df = pd.read_csv(summary_path)

    models = ["chow_lin", "elastic_net", "xgboost", "mlp"]
    model_labels = ["Chow-Lin", "Elastic Net", "XGBoost", "MLP"]
    countries = ["united_states", "germany", "united_kingdom", "china"]
    country_labels = ["US", "Germany", "UK", "China"]
    lags = [0, 1, 2]

    fig, axes = plt.subplots(1, 4, figsize=(14, 4), sharey=True)

    for idx, (country, clabel) in enumerate(zip(countries, country_labels)):
        ax = axes[idx]
        data = np.zeros((len(models), len(lags)))
        for i, model in enumerate(models):
            for j, lag in enumerate(lags):
                row = df[(df["Country"] == country) &
                         (df["Model"] == model) &
                         (df["Lag"] == lag)]
                if len(row) > 0:
                    data[i, j] = row["R2"].values[0]

        bwrb = LinearSegmentedColormap.from_list(
            "bwr_pub", ["#1d4ed8", "#ffffff", "#dc2626"]
        )
        im = ax.imshow(data, cmap=bwrb, aspect="auto",
                       vmin=-1.2, vmax=1.0)
        ax.set_xticks(range(len(lags)))
        ax.set_xticklabels([f"Lag {l}" for l in lags])
        ax.set_title(clabel, fontsize=12)

        if idx == 0:
            ax.set_yticks(range(len(models)))
            ax.set_yticklabels(model_labels)

        # Annotate cells
        for i in range(len(models)):
            for j in range(len(lags)):
                val = data[i, j]
                color = "white" if abs(val) > 0.6 else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=9, color=color)

    fig.colorbar(im, ax=axes, label="$R^2$", shrink=0.8, pad=0.02)
    fig.suptitle("Out-of-Sample $R^2$ by Country, Model, and Lag",
                 fontsize=14, y=1.02)
    fig.savefig(f"{outdir}/fig_r2_heatmap.pdf")
    plt.close(fig)
    print("  Saved: fig_r2_heatmap")


# ── Figure 6: US Chow-Lin vs Elastic Net degradation ─────
def fig_us_lag_degradation(summary_path, outdir):
    """Show Chow-Lin degradation vs Elastic Net stability across lags."""
    df = pd.read_csv(summary_path)
    us = df[df["Country"] == "united_states"]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    lags = [0, 1, 2]
    all_r2 = []

    for model, label, color, ls in [
        ("chow_lin", "Chow-Lin", COLORS["chow_lin"], "-"),
        ("elastic_net", "Elastic Net", COLORS["elastic_net"], "-"),
        ("xgboost", "XGBoost", COLORS["xgboost"], "--"),
        ("mlp", "MLP", COLORS["mlp"], "--"),
    ]:
        r2s = []
        for lag in lags:
            row = us[(us["Model"] == model) & (us["Lag"] == lag)]
            r2s.append(row["R2"].values[0] if len(row) > 0 else np.nan)
        all_r2.extend([v for v in r2s if not np.isnan(v)])
        ax.plot(lags, r2s, color=color, linestyle=ls,
                marker="o", linewidth=2, markersize=7, label=label)

    ax.axhline(0, color="gray", linewidth=0.5, linestyle=":")
    ax.set_xlabel("Number of Quarterly Lags")
    ax.set_ylabel("Out-of-Sample $R^2$")
    ax.set_xticks(lags)
    ax.set_title("United States: $R^2$ by Lag Specification")
    ax.legend(frameon=False)
    # Expand lower bound so negative R^2 (e.g., MLP) is never clipped.
    y_min = min(all_r2) if all_r2 else -1.3
    ax.set_ylim(min(-1.3, y_min - 0.1), 1.0)
    fig.savefig(f"{outdir}/fig_us_lag_degradation.pdf")
    plt.close(fig)
    print("  Saved: fig_us_lag_degradation")


# ── Figure 7: SHAP Variable Importance Comparison ─────────
def fig_shap_comparison(shap_en_path, shap_xgb_path, outdir):
    """Side-by-side SHAP importance: Elastic Net vs XGBoost (US)."""
    en = pd.read_csv(shap_en_path)
    xgb = pd.read_csv(shap_xgb_path)

    def clean_name(s):
        s = str(s).replace("L1_", "L1: ").replace("^GSPC_vol", "S&P 500 Vol.")
        s = s.replace("^GSPC", "S&P 500").replace("_", " ")
        s = s.replace("working hour manuf", "Mfg. Hours")
        s = s.replace("Fed Funds", "Fed Funds Rate")
        s = s.replace("Intr 10Y", "10Y Treasury")
        s = s.replace("Moody aaa", "Moody AAA")
        s = s.replace("Ids Prd", "Ind. Production")
        s = s.replace("R Csump", "Real Consump.")
        s = s.replace("All Emp", "Nonfarm Payroll")
        s = s.replace("Labor partic", "Labor Part.")
        s = s.replace("NetExports", "Net Exports")
        return s

    en_top = en.head(10).copy()
    xgb_top = xgb.head(10).copy()
    en_top["Feature"] = en_top["Feature"].apply(clean_name)
    xgb_top["Feature"] = xgb_top["Feature"].apply(clean_name)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    ax1.barh(
        range(len(en_top) - 1, -1, -1),
        en_top["Mean_Abs_SHAP"].values,
        color=COLORS["elastic_net"],
        alpha=0.85,
        height=0.7,
    )
    ax1.set_yticks(range(len(en_top) - 1, -1, -1))
    ax1.set_yticklabels(en_top["Feature"].values)
    ax1.set_xlabel("Mean |SHAP|")
    ax1.set_title("Elastic Net")

    ax2.barh(
        range(len(xgb_top) - 1, -1, -1),
        xgb_top["Mean_Abs_SHAP"].values,
        color=COLORS["xgboost"],
        alpha=0.85,
        height=0.7,
    )
    ax2.set_yticks(range(len(xgb_top) - 1, -1, -1))
    ax2.set_yticklabels(xgb_top["Feature"].values)
    ax2.set_xlabel("Mean |SHAP|")
    ax2.set_title("XGBoost")

    fig.suptitle("United States: SHAP Variable Importance Comparison", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(f"{outdir}/fig_shap_comparison.pdf")
    plt.close(fig)
    print("  Saved: fig_shap_comparison")


# ── Main ──────────────────────────────────────────────────
def main():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    parser = argparse.ArgumentParser()
    parser.add_argument("--us_monthly", default=None,
                        help="US Elastic Net lag=1 monthly GDP CSV")
    parser.add_argument("--uk_monthly", default=None,
                        help="UK Chow-Lin lag=0 monthly GDP CSV")
    parser.add_argument("--koop", default=None,
                        help="Koop et al. monthly GDP CSV")
    parser.add_argument("--ons", default=None,
                        help="ONS monthly GDP CSV")
    parser.add_argument("--summary", default=None,
                        help="summary_all.csv with all metrics")
    parser.add_argument("--shap_en", default=None,
                        help="SHAP CSV for Elastic Net (US)")
    parser.add_argument("--shap_xgb", default=None,
                        help="SHAP CSV for XGBoost (US)")
    parser.add_argument("--outdir", default=os.path.join(project_root, "figures"),
                        help="Output directory for figures")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # Auto-resolve default inputs so "Run" button works without args.
    us_monthly = resolve_input_path(
        args.us_monthly,
        [
            os.path.join(project_root, "results", "united_states", "monthly_gdp_US_elastic_net_lag1.csv"),
            os.path.join(project_root, "results", "united_states", "monthly_gdp_elastic_net_lag1.csv"),
        ],
        "us_monthly",
    )
    uk_monthly = resolve_input_path(
        args.uk_monthly,
        [
            os.path.join(project_root, "results", "united_kingdom", "monthly_gdp_UK_chow_lin_lag0.csv"),
            os.path.join(project_root, "results", "united_kingdom", "monthly_gdp_chow_lin_lag0.csv"),
        ],
        "uk_monthly",
    )
    koop_path = resolve_input_path(
        args.koop,
        [os.path.join(project_root, "data", "raw", "koop_montly_gdp.csv")],
        "koop",
    )
    ons_path = resolve_input_path(
        args.ons,
        [os.path.join(project_root, "data", "raw", "ons_montly_gdp.csv")],
        "ons",
    )
    summary_path = resolve_input_path(
        args.summary,
        [os.path.join(project_root, "results", "summary_all.csv")],
        "summary",
    )
    shap_en_path = resolve_optional_path(
        args.shap_en,
        [
            os.path.join(project_root, "results", "united_states", "shap_US_elastic_net_lag1.csv"),
            os.path.join(project_root, "results", "united_states", "shap_US_elastic_net_lag0.csv"),
            os.path.join(project_root, "results", "united_states", "shap_elastic_net_lag1.csv"),
        ],
    )
    shap_xgb_path = resolve_optional_path(
        args.shap_xgb,
        [
            os.path.join(project_root, "results", "united_states", "shap_US_xgboost_lag0.csv"),
            os.path.join(project_root, "results", "united_states", "shap_US_xgboost_lag1.csv"),
            os.path.join(project_root, "results", "united_states", "shap_xgboost_lag0.csv"),
        ],
    )

    print("Generating figures...")
    fig_us_monthly_level(us_monthly, args.outdir)
    fig_us_monthly_growth(us_monthly, args.outdir)
    fig_us_benchmark(us_monthly, koop_path, args.outdir)
    fig_uk_benchmark(uk_monthly, ons_path, args.outdir)
    fig_r2_heatmap(summary_path, args.outdir)
    fig_us_lag_degradation(summary_path, args.outdir)
    if shap_en_path and shap_xgb_path:
        fig_shap_comparison(shap_en_path, shap_xgb_path, args.outdir)
    else:
        print("  Skipped: fig_shap_comparison (missing SHAP input files)")
    print("Done.")


if __name__ == "__main__":
    main()
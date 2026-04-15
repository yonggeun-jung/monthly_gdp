"""
Data Preprocessing Module (Algorithm 1 & 2)
-------------------------------------------
Handles seasonal adjustment, ADF testing, log-differencing,
quarterly aggregation, train/test splitting, and standardization.
"""

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import STL
from sklearn.preprocessing import StandardScaler
from dataclasses import dataclass, field
from typing import Optional


def seasonal_adjust(
    df: pd.DataFrame,
    cols: list,
    period: int = 12,
) -> pd.DataFrame:
    """
    Apply STL seasonal adjustment to specified columns.

    Uses statsmodels STL (Seasonal-Trend decomposition using LOESS).
    Replaces each column with its seasonally adjusted version
    (trend + residual, removing the seasonal component).

    Parameters
    ----------
    df : pd.DataFrame
        Input data with a DatetimeIndex or DATE column.
    cols : list
        Columns to seasonally adjust.
    period : int
        Seasonal period (12 for monthly data).

    Returns
    -------
    pd.DataFrame
        Copy of df with specified columns replaced by SA versions.
    """
    result = df.copy()
    for col in cols:
        series = pd.to_numeric(result[col], errors="coerce")
        valid = series.dropna()
        if len(valid) < 2 * period:
            continue
        try:
            stl = STL(valid, period=period, robust=True)
            decomp = stl.fit()
            # SA = original - seasonal = trend + residual
            sa = decomp.trend + decomp.resid
            result.loc[sa.index, col] = sa
        except Exception as e:
            print(f"  Warning: STL failed for {col}: {e}")
    return result


@dataclass
class ProcessedData:
    """Container for all processed data needed by models."""
    # Quarterly (for training)
    X_q_train: pd.DataFrame
    Y_q_train: pd.Series
    X_q_test: pd.DataFrame
    Y_q_test: pd.Series
    X_q_train_scaled: pd.DataFrame
    X_q_test_scaled: pd.DataFrame
    # Monthly (for disaggregation)
    X_prime_m_df: pd.DataFrame
    X_m_scaled: pd.DataFrame
    # Quarterly levels (for level reconstruction)
    Y_q_levels: pd.Series
    Y_q_processed: pd.Series
    # Full quarterly (for expanding window)
    X_q_processed: pd.DataFrame
    # Metadata
    log_diff_cols: list
    diff_cols: list
    non_log_cols: list
    scaler: StandardScaler
    x_cols: list
    x_cols_with_lags: list
    adf_results: pd.DataFrame


def run_adf_tests(df: pd.DataFrame, x_cols: list) -> pd.DataFrame:
    """Run Augmented Dickey-Fuller test on each explanatory variable."""
    rows = []
    for col in x_cols:
        series = pd.to_numeric(df[col], errors="coerce").dropna()
        if not series.empty:
            result = adfuller(series)
            rows.append({
                "Variable": col,
                "ADF Statistic": result[0],
                "p-value": result[1],
                "Stationary at 5%": result[1] < 0.05,
            })
        else:
            rows.append({
                "Variable": col,
                "ADF Statistic": np.nan,
                "p-value": np.nan,
                "Stationary at 5%": False,
            })
    return pd.DataFrame(rows).set_index("Variable").sort_values("p-value")


def transform_monthly(
    df: pd.DataFrame,
    x_cols: list,
    log_diff_cols: list,
    diff_cols: list = None,
) -> pd.DataFrame:
    """
    Apply transformations to monthly explanatory variables.
    - log_diff_cols: log-difference (for trending positive variables)
    - diff_cols: first-difference without log (for rates like interest rates)
    - others: keep as levels
    Returns X'_m (transformed monthly explanatory variables).
    """
    if diff_cols is None:
        diff_cols = []
    X_prime = df[["DATE", "quarter"]].copy()
    for col in x_cols:
        if col in log_diff_cols:
            vals = df[col].copy()
            if (vals <= 0).any():
                vals = vals.replace(0, np.nan)
            X_prime[col] = np.log(vals).diff()
        elif col in diff_cols:
            X_prime[col] = df[col].diff()
        else:
            X_prime[col] = df[col].copy()
    return X_prime


def aggregate_quarterly(
    X_prime_m_df: pd.DataFrame,
    log_diff_cols: list,
    non_log_cols: list,
    diff_cols: list = None,
) -> pd.DataFrame:
    """
    Aggregate monthly X' to quarterly frequency.
    Log-differenced variables: quarterly sum (cumulative growth).
    First-differenced variables: quarterly sum (cumulative change).
    Level variables: quarterly mean.
    """
    if diff_cols is None:
        diff_cols = []
    groups = X_prime_m_df.groupby("quarter")
    parts = []
    # Both log-diff and diff cols are summed
    sum_cols = [c for c in log_diff_cols + diff_cols
                if c in X_prime_m_df.columns]
    if sum_cols:
        parts.append(groups[sum_cols].sum())
    if non_log_cols:
        parts.append(groups[non_log_cols].mean())
    return pd.concat(parts, axis=1).sort_index()


def add_lags(
    df: pd.DataFrame,
    cols: list,
    max_lags: int = 3,
    prefix: str = "L",
) -> tuple[pd.DataFrame, list]:
    """
    Add lagged columns to a DataFrame.

    For quarterly data, lag k means k quarters back.
    For monthly data, lag k means k months back.

    Parameters
    ----------
    df : pd.DataFrame
        Input data (quarterly or monthly).
    cols : list
        Columns to create lags for.
    max_lags : int
        Number of lags (1 to max_lags).
    prefix : str
        Prefix for lag column names.

    Returns
    -------
    df_with_lags : pd.DataFrame
        Original + lagged columns.
    lag_col_names : list
        Names of newly created lag columns.
    """
    result = df.copy()
    new_cols = []
    for lag in range(1, max_lags + 1):
        for col in cols:
            lag_name = f"{prefix}{lag}_{col}"
            result[lag_name] = df[col].shift(lag)
            new_cols.append(lag_name)
    return result, new_cols


def prepare_target(
    df: pd.DataFrame, Y_col: str
) -> tuple[pd.Series, pd.Series]:
    """
    Extract quarterly GDP levels and compute log-difference growth rates.
    Returns (Y_q_levels, Y_q_processed).
    """
    y_q_levels_df = (
        df[df[Y_col].notna()][["quarter", Y_col]]
        .drop_duplicates(subset="quarter", keep="first")
        .set_index("quarter")
        .sort_index()
    )
    Y_q_levels = y_q_levels_df[Y_col]
    Y_q_processed = np.log(Y_q_levels).diff().dropna()
    return Y_q_levels, Y_q_processed


def preprocess(
    master_csv: str,
    target_col: str,
    log_diff_cols: list,
    diff_cols: list = None,
    train_ratio: float = 0.5,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    data_dir: str = "data/raw",
    max_lags: int = 0,
    seasonal_adjust_cols: Optional[list] = None,
) -> ProcessedData:
    """
    Full preprocessing pipeline (Algorithm 1 & 2).

    Parameters
    ----------
    master_csv : str
        Filename of the country's master CSV.
    target_col : str
        Name of the GDP column.
    log_diff_cols : list
        Columns to log-difference (trending positive variables).
    diff_cols : list or None
        Columns to first-difference without log (e.g. interest rates).
    train_ratio : float
        Fraction of quarters for training (used for default split).
    start_date : str or None
        Optional start date filter (e.g. "1994-01-01" for China).
    end_date : str or None
        Optional end date filter, inclusive (e.g. "2024-12-31" for 2024Q4).
    data_dir : str
        Directory containing the master CSV.
    max_lags : int
        Number of quarterly lags to add (0 = contemporaneous only).
    seasonal_adjust_cols : list or None
        Columns to seasonally adjust via STL before processing.
        If "all", adjusts all explanatory variables.

    Returns
    -------
    ProcessedData
    """
    if diff_cols is None:
        diff_cols = []

    import os
    filepath = os.path.join(data_dir, master_csv)
    df = pd.read_csv(filepath)
    df["DATE"] = pd.to_datetime(df["DATE"])

    if start_date is not None:
        df = df[df["DATE"] >= pd.to_datetime(start_date)]
    if end_date is not None:
        df = df[df["DATE"] <= pd.to_datetime(end_date)]
    df = df.reset_index(drop=True)

    df["quarter"] = df["DATE"].dt.to_period("Q")

    # Identify explanatory variables
    exclude = {"DATE", "quarter", target_col}
    x_cols = [c for c in df.columns if c not in exclude]
    transformed_cols = set(log_diff_cols) | set(diff_cols)
    non_log_cols = [c for c in x_cols if c not in transformed_cols]

    # Seasonal adjustment (before ADF and transformations)
    if seasonal_adjust_cols is not None:
        if seasonal_adjust_cols == "all":
            sa_cols = x_cols
        else:
            sa_cols = [c for c in seasonal_adjust_cols if c in x_cols]
        if sa_cols:
            print(f"   Applying STL seasonal adjustment to {len(sa_cols)} columns...")
            df = seasonal_adjust(df, sa_cols, period=12)

    # ADF tests (informational)
    adf_results = run_adf_tests(df, x_cols)

    # Transform monthly variables
    X_prime_m_df = transform_monthly(df, x_cols, log_diff_cols, diff_cols)

    # Aggregate to quarterly
    X_q_agg = aggregate_quarterly(X_prime_m_df, log_diff_cols, non_log_cols, diff_cols)

    # Add quarterly lags if requested
    x_cols_with_lags = list(X_q_agg.columns)
    lag_col_names = []
    if max_lags > 0:
        X_q_agg, lag_col_names = add_lags(
            X_q_agg, cols=list(X_q_agg.columns), max_lags=max_lags
        )
        x_cols_with_lags = list(X_q_agg.columns)

    # Prepare target
    Y_q_levels, Y_q_processed = prepare_target(df, target_col)

    # Align X and Y, drop NaN rows from lagging
    common_idx = X_q_agg.index.intersection(Y_q_processed.index)
    X_q_processed = X_q_agg.loc[common_idx].dropna()
    Y_q_processed = Y_q_processed.loc[X_q_processed.index]

    # Identify which columns need standardization
    # (non-log original cols + their lags)
    non_log_all = [c for c in X_q_processed.columns
                   if c in non_log_cols
                   or any(c.endswith(f"_{nl}") for nl in non_log_cols)]

    # Train/test split (temporal, no shuffling)
    split_idx = int(len(X_q_processed) * train_ratio)
    X_q_train = X_q_processed.iloc[:split_idx]
    Y_q_train = Y_q_processed.iloc[:split_idx]
    X_q_test = X_q_processed.iloc[split_idx:]
    Y_q_test = Y_q_processed.iloc[split_idx:]

    # Standardize non-log columns (fit on train only)
    # Quarterly scaler: includes lag columns
    q_scaler = StandardScaler()
    X_q_train_scaled = X_q_train.copy()
    X_q_test_scaled = X_q_test.copy()
    cols_to_scale = [c for c in non_log_all if c in X_q_train.columns]
    if cols_to_scale:
        X_q_train_scaled[cols_to_scale] = q_scaler.fit_transform(
            X_q_train[cols_to_scale]
        )
        X_q_test_scaled[cols_to_scale] = q_scaler.transform(
            X_q_test[cols_to_scale]
        )

    # Monthly scaler: original columns only (no lags)
    m_scaler = StandardScaler()
    monthly_cols = [c for c in x_cols if c in X_prime_m_df.columns]
    X_m_scaled = X_prime_m_df[monthly_cols].copy()
    monthly_scale_cols = [c for c in non_log_cols if c in X_m_scaled.columns]
    if monthly_scale_cols:
        # Fit on the training period of monthly data
        train_end_q = X_q_train.index[-1]
        train_mask = X_prime_m_df["quarter"] <= train_end_q
        m_scaler.fit(X_m_scaled.loc[train_mask, monthly_scale_cols])
        X_m_scaled[monthly_scale_cols] = m_scaler.transform(
            X_m_scaled[monthly_scale_cols]
        )

    # Add monthly lags if quarterly lags were used
    # Quarterly lag k = monthly shift of 3*k
    if max_lags > 0:
        for lag in range(1, max_lags + 1):
            monthly_shift = lag * 3
            for col in monthly_cols:
                lag_name = f"L{lag}_{col}"
                X_m_scaled[lag_name] = X_m_scaled[col].shift(monthly_shift)

    # Ensure monthly columns match quarterly model columns exactly
    q_cols = list(X_q_train_scaled.columns)
    X_m_scaled = X_m_scaled.reindex(columns=q_cols)

    return ProcessedData(
        X_q_train=X_q_train,
        Y_q_train=Y_q_train,
        X_q_test=X_q_test,
        Y_q_test=Y_q_test,
        X_q_train_scaled=X_q_train_scaled,
        X_q_test_scaled=X_q_test_scaled,
        X_prime_m_df=X_prime_m_df,
        X_m_scaled=X_m_scaled,
        Y_q_levels=Y_q_levels,
        Y_q_processed=Y_q_processed,
        X_q_processed=X_q_processed,
        log_diff_cols=log_diff_cols,
        diff_cols=diff_cols,
        non_log_cols=non_log_cols,
        scaler=q_scaler,
        x_cols=x_cols,
        x_cols_with_lags=x_cols_with_lags,
        adf_results=adf_results,
    )
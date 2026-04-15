"""
Temporal Reconciliation with Mariano-Murasawa Constraints
---------------------------------------------------------
Implements the log-linear approximation of Mariano and Murasawa
(2003, 2010) for relating monthly and quarterly growth rates:

    Y_q = (1/3) y_m + (2/3) y_{m-1} + y_{m-2} + (2/3) y_{m-3} + (1/3) y_{m-4}

where m is the last month of quarter q.

Given preliminary monthly estimates s from the ML model, we solve:

    min  || y - s ||^2     s.t.  M y = z

Closed-form: y* = s + M'(M M')^{-1}(z - M s)
"""

import numpy as np
import pandas as pd
from scipy import linalg

MA5_WEIGHTS = np.array([1/3, 2/3, 1.0, 2/3, 1/3])


def build_constraint_matrix(T: int, quarter_end_indices: list) -> np.ndarray:
    """
    Build Q x T constraint matrix M with MA(5) weights.

    Parameters
    ----------
    T : int
        Total number of months.
    quarter_end_indices : list of int
        0-based index of the last month of each quarter.

    Returns
    -------
    M : ndarray, shape (Q, T)
    """
    Q = len(quarter_end_indices)
    M = np.zeros((Q, T))
    for row, m in enumerate(quarter_end_indices):
        for lag, w in enumerate(MA5_WEIGHTS):
            col = m - lag
            if 0 <= col < T:
                M[row, col] = w
    return M


def reconcile_mariano_murasawa(
    monthly_signal: np.ndarray,
    quarterly_growth: np.ndarray,
    quarter_end_indices: list,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Reconcile monthly estimates with observed quarterly growth
    using Mariano-Murasawa MA(5) constraints.

    Parameters
    ----------
    monthly_signal : ndarray, shape (T,)
        Preliminary monthly growth estimates from the model.
    quarterly_growth : ndarray, shape (Q,)
        Observed quarterly log-difference growth rates.
    quarter_end_indices : list of int
        Last-month positions for each quarter.

    Returns
    -------
    y : ndarray, shape (T,)
        Reconciled monthly growth rates.
    k_factors : ndarray, shape (Q,)
        Ratio of actual to pre-reconciliation quarterly value
        (diagnostic for adjustment magnitude).
    """
    T = len(monthly_signal)
    Q = len(quarterly_growth)
    s = monthly_signal.copy()

    M = build_constraint_matrix(T, quarter_end_indices)

    # Drop quarters without enough history for full MA(5)
    valid = np.array([quarter_end_indices[i] >= 4 for i in range(Q)])
    M_v = M[valid]
    z_v = quarterly_growth[valid]

    # Constraint violation of preliminary estimates
    residual = z_v - M_v @ s

    # Closed-form correction
    MMt = M_v @ M_v.T
    try:
        correction = M_v.T @ linalg.solve(MMt, residual, assume_a="pos")
    except linalg.LinAlgError:
        correction = M_v.T @ np.linalg.lstsq(MMt, residual, rcond=None)[0]

    y = s + correction

    # Diagnostic: adjustment factors
    pre = M_v @ s
    k_factors = np.full(Q, np.nan)
    for i, idx in enumerate(np.where(valid)[0]):
        if abs(pre[i]) > 1e-12:
            k_factors[idx] = z_v[i] / pre[i]

    return y, k_factors


def recover_monthly_levels(
    adjusted_growth: np.ndarray,
    base_level: float,
) -> np.ndarray:
    """Recover GDP levels: level_m = level_{m-1} * exp(y_m)."""
    T = len(adjusted_growth)
    levels = np.full(T, np.nan)
    current = base_level
    for t in range(T):
        g = adjusted_growth[t]
        if np.isnan(g):
            continue
        current = current * np.exp(g)
        levels[t] = current
    return levels


def full_disaggregation(
    model_predict_fn,
    X_m_scaled,
    X_prime_m_df: pd.DataFrame,
    Y_q_processed: pd.Series,
    Y_q_levels: pd.Series,
) -> pd.DataFrame:
    """
    End-to-end pipeline: signal -> MA(5) reconciliation -> levels.

    Returns DataFrame with: raw_signal, adjusted_growth, monthly_level, k_factor
    """
    dates = pd.to_datetime(X_prime_m_df["DATE"].values)
    quarters = dates.to_period("Q")

    # Step 1: raw monthly signal
    # Some months have NaN features (e.g. first row after differencing,
    # or initial rows when lags are used).  Linear models and MLPs
    # propagate NaN, so we predict only on valid rows and fill the rest
    # with 0 (neutral growth) before reconciliation.
    X_arr = X_m_scaled.values if hasattr(X_m_scaled, 'values') else X_m_scaled
    valid_mask = ~np.isnan(X_arr).any(axis=1)
    raw = np.zeros(len(X_arr))
    if valid_mask.any():
        raw[valid_mask] = np.asarray(
            model_predict_fn(X_arr[valid_mask])
        ).flatten()

    # Identify quarter-end months aligned with Y_q
    unique_q = sorted(set(quarters))
    q_end_idx, q_vals = [], []
    for q in unique_q:
        positions = np.where(quarters == q)[0]
        if len(positions) == 0:
            continue
        if q in Y_q_processed.index:
            q_end_idx.append(positions[-1])
            q_vals.append(Y_q_processed.loc[q])

    # Step 2: MA(5) reconciliation
    adjusted, k_factors = reconcile_mariano_murasawa(
        raw, np.array(q_vals), q_end_idx,
    )

    # Step 3: levels
    # Anchor the level reconstruction at the first month whose features
    # were actually valid (i.e. not filled with 0 due to NaN).
    # This prevents the anchor from shifting to the very start of the
    # series when NaN rows are replaced with neutral growth.
    first_valid = np.where(valid_mask)[0]
    if len(first_valid) > 0:
        first_q = dates[first_valid[0]].to_period("Q")
        base_q = first_q - 1
        base_level = (Y_q_levels.loc[base_q]
                      if base_q in Y_q_levels.index
                      else Y_q_levels.iloc[0])
    else:
        base_level = Y_q_levels.iloc[0]

    levels = recover_monthly_levels(adjusted, base_level)

    # Map k_factors to monthly series for diagnostics
    k_series = np.full(len(dates), np.nan)
    for i, idx in enumerate(q_end_idx):
        if i < len(k_factors) and not np.isnan(k_factors[i]):
            mask = quarters == quarters[idx]
            k_series[mask] = k_factors[i]

    return pd.DataFrame({
        "raw_signal": raw,
        "adjusted_growth": adjusted,
        "monthly_level": levels,
        "k_factor": k_series,
    }, index=dates)
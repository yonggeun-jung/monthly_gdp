"""
Evaluation Metrics
------------------
Quarterly forecast accuracy metrics and Diebold-Mariano test.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy import stats


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute all standard forecast accuracy metrics."""
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()

    if len(y_true) == 0 or len(y_pred) == 0:
        return {k: np.nan for k in
                ["MSE", "RMSE", "MAE", "R2", "Correlation",
                 "Sign Accuracy", "sMAPE", "Theil U1"]}

    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    corr = np.corrcoef(y_true, y_pred)[0, 1] if len(y_true) > 1 else np.nan
    sign_acc = np.mean(np.sign(y_pred) == np.sign(y_true))
    smape_val = _smape(y_true, y_pred)
    theil = _theil_u1(y_true, y_pred)

    return {
        "MSE": mse,
        "RMSE": rmse,
        "MAE": mae,
        "R2": r2,
        "Correlation": corr,
        "Sign Accuracy": sign_acc,
        "sMAPE": smape_val,
        "Theil U1": theil,
    }


def _smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    num = 2.0 * np.abs(y_pred - y_true)
    den = np.abs(y_true) + np.abs(y_pred)
    ratio = np.divide(num, den, out=np.zeros_like(num), where=den != 0)
    return float(np.mean(ratio) * 100)


def _theil_u1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    rms_a = np.sqrt(np.mean(y_true ** 2))
    rms_p = np.sqrt(np.mean(y_pred ** 2))
    denom = rms_a + rms_p
    return float(rmse / denom) if denom > 0 else np.nan


def diebold_mariano_test(
    y_true: np.ndarray,
    y_pred_1: np.ndarray,
    y_pred_2: np.ndarray,
    loss: str = "squared",
    h: int = 1,
) -> dict:
    """
    Diebold-Mariano test for equal predictive accuracy.

    H0: E[d_t] = 0  (equal predictive accuracy)
    H1: E[d_t] != 0

    Parameters
    ----------
    y_true : array-like
        Actual values.
    y_pred_1, y_pred_2 : array-like
        Predictions from model 1 and model 2.
    loss : str
        "squared" for squared error, "absolute" for absolute error.
    h : int
        Forecast horizon (for HAC variance adjustment).

    Returns
    -------
    dict with DM statistic, p-value, and interpretation.
    """
    y_true = np.asarray(y_true).flatten()
    y_pred_1 = np.asarray(y_pred_1).flatten()
    y_pred_2 = np.asarray(y_pred_2).flatten()

    if loss == "squared":
        e1 = (y_true - y_pred_1) ** 2
        e2 = (y_true - y_pred_2) ** 2
    elif loss == "absolute":
        e1 = np.abs(y_true - y_pred_1)
        e2 = np.abs(y_true - y_pred_2)
    else:
        raise ValueError(f"Unknown loss: {loss}")

    d = e1 - e2
    n = len(d)
    d_bar = np.mean(d)

    # HAC variance (Newey-West with bandwidth h-1)
    gamma_0 = np.mean((d - d_bar) ** 2)
    gamma_sum = 0.0
    for k in range(1, h):
        gamma_k = np.mean((d[k:] - d_bar) * (d[:-k] - d_bar))
        gamma_sum += 2.0 * gamma_k
    var_d = (gamma_0 + gamma_sum) / n

    if var_d <= 0:
        return {"DM_stat": np.nan, "p_value": np.nan, "note": "non-positive variance"}

    dm_stat = d_bar / np.sqrt(var_d)
    p_value = 2.0 * (1.0 - stats.norm.cdf(np.abs(dm_stat)))

    return {
        "DM_stat": float(dm_stat),
        "p_value": float(p_value),
        "mean_loss_diff": float(d_bar),
        "model1_better": d_bar < 0,
    }
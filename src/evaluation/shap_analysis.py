"""
SHAP-Based Model Interpretability
----------------------------------
Unified interpretability framework across all model classes,
replacing inconsistent variable importance methods.

Addresses Referee 1's recommendation to use Shapley values
for coherent cross-model comparison.
"""

import numpy as np
import pandas as pd


def compute_shap_values(model, X, feature_names, model_type="generic"):
    """
    Compute SHAP values for any fitted model.

    Parameters
    ----------
    model : fitted model object
        The trained model (sklearn-compatible or keras).
    X : ndarray, shape (n_samples, n_features)
        Input data for explanation.
    feature_names : list of str
        Feature names.
    model_type : str
        One of "tree" (XGBoost), "linear" (ElasticNet),
        "kernel" (MLP), or "generic" (auto-detect).

    Returns
    -------
    shap_values : ndarray, shape (n_samples, n_features)
    expected_value : float
        Base value (average prediction).
    """
    import shap

    if model_type == "tree":
        explainer = shap.TreeExplainer(model)
        sv = explainer.shap_values(X)
        ev = explainer.expected_value
    elif model_type == "linear":
        explainer = shap.LinearExplainer(model, X)
        sv = explainer.shap_values(X)
        ev = explainer.expected_value
    elif model_type == "kernel":
        # For neural networks: use KernelExplainer with background sample
        bg = shap.sample(X, min(50, len(X)))
        explainer = shap.KernelExplainer(model.predict, bg)
        sv = explainer.shap_values(X, nsamples=100)
        ev = explainer.expected_value
    else:
        # Auto-detect
        try:
            explainer = shap.Explainer(model, X)
            sv = explainer(X).values
            ev = explainer.expected_value
        except Exception:
            bg = shap.sample(X, min(50, len(X)))
            explainer = shap.KernelExplainer(model.predict, bg)
            sv = explainer.shap_values(X, nsamples=100)
            ev = explainer.expected_value

    if isinstance(sv, list):
        sv = sv[0]

    return np.asarray(sv), float(np.mean(ev)) if hasattr(ev, '__len__') else float(ev)


def shap_importance_table(
    shap_values: np.ndarray,
    feature_names: list,
) -> pd.DataFrame:
    """
    Compute mean |SHAP| importance ranking.

    Returns sorted DataFrame with columns:
        Feature, Mean_Abs_SHAP, Rank
    """
    mean_abs = np.mean(np.abs(shap_values), axis=0)
    df = pd.DataFrame({
        "Feature": feature_names,
        "Mean_Abs_SHAP": mean_abs,
    }).sort_values("Mean_Abs_SHAP", ascending=False).reset_index(drop=True)
    df["Rank"] = range(1, len(df) + 1)
    return df


def shap_temporal_analysis(
    shap_values: np.ndarray,
    feature_names: list,
    dates: pd.Index,
    top_k: int = 5,
) -> pd.DataFrame:
    """
    Analyze how SHAP contributions evolve over time.
    Useful for tracing nonlinearities around GFC/COVID.

    Returns DataFrame with dates as index and top-k features as columns,
    showing the SHAP value at each time step.
    """
    df = pd.DataFrame(shap_values, columns=feature_names, index=dates)
    # Select top-k by overall importance
    mean_abs = np.mean(np.abs(shap_values), axis=0)
    top_features = [feature_names[i] for i in np.argsort(mean_abs)[::-1][:top_k]]
    return df[top_features]

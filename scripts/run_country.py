"""
run_country.py
--------------
Main entry point with expanding window out-of-sample evaluation.
 
Usage:
    python scripts/run_country.py --country china --model mlp
    python scripts/run_country.py --country china --model all --lags 2
    python scripts/run_country.py --country all --model all --lags 2
"""
 
import argparse
import os
import sys
import json
import yaml
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
 
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
 
from src.data.preprocess import preprocess, add_lags
from src.evaluation.metrics import compute_metrics, diebold_mariano_test
from src.evaluation.denton import full_disaggregation
 
MODEL_REGISTRY = {
    "mlp": ("src.models.mlp", "MLPModel"),
    "xgboost": ("src.models.xgboost_model", "XGBoostModel"),
    "elastic_net": ("src.models.elastic_net", "ElasticNetModel"),
    "chow_lin": ("src.models.chow_lin", "ChowLinModel"),
}
COUNTRY_NAMES = ["china", "united_states", "united_kingdom", "germany"]
# ISO-style tags for result filenames (ew_CN_mlp_lag0.csv, etc.)
COUNTRY_CODES = {
    "china": "CN",
    "germany": "DE",
    "united_kingdom": "UK",
    "united_states": "US",
}
 
 
def load_config(config_path=None):
    if config_path is None:
        config_path = os.path.join(PROJECT_ROOT, "config", "countries.yaml")
    with open(config_path) as f:
        return yaml.safe_load(f)
 
 
def get_model(model_name, seed=42, **kwargs):
    module_path, class_name = MODEL_REGISTRY[model_name]
    import importlib
    mod = importlib.import_module(module_path)
    return getattr(mod, class_name)(seed=seed, **kwargs)
 
 
def expanding_window_eval(
    X_q: pd.DataFrame,
    Y_q: pd.Series,
    model_name: str,
    model_kwargs: dict,
    non_log_cols: list,
    initial_ratio: float = 0.6,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Expanding window (recursive) out-of-sample evaluation.

    Architecture is selected via hyperparameter search on the initial
    training window and held fixed for all subsequent windows.
    Only model weights are re-estimated as the window expands.
    """
    N = len(X_q)
    init_size = max(int(N * initial_ratio), 10)
    results = []

    cols_to_scale = [c for c in non_log_cols if c in X_q.columns]
    best_hp = None  # Fixed after first window

    for t in range(init_size, N):
        # Training window: [0, t)
        X_train = X_q.iloc[:t]
        Y_train = Y_q.iloc[:t]

        # Test: single quarter at position t
        X_test = X_q.iloc[t : t + 1]
        Y_test = Y_q.iloc[t : t + 1]

        # Scale (fit on training window only)
        scaler = StandardScaler()
        X_train_s = X_train.copy()
        X_test_s = X_test.copy()
        if cols_to_scale:
            X_train_s[cols_to_scale] = scaler.fit_transform(X_train[cols_to_scale])
            X_test_s[cols_to_scale] = scaler.transform(X_test[cols_to_scale])

        # Split training into train/val for neural nets
        val_size = max(int(len(X_train_s) * 0.2), 1)
        X_tr = X_train_s.iloc[:-val_size].values
        y_tr = Y_train.iloc[:-val_size].values
        X_vl = X_train_s.iloc[-val_size:].values
        y_vl = Y_train.iloc[-val_size:].values

        model = get_model(model_name, seed=seed, **model_kwargs)
        try:
            if best_hp is not None and hasattr(model, "fit_fixed"):
                # Subsequent windows: fixed architecture, retrain only
                model.fit_fixed(
                    X_train=X_tr, y_train=y_tr,
                    X_val=X_vl, y_val=y_vl,
                    hp_dict=best_hp,
                )
            else:
                # First window: full hyperparameter search
                best_hp = model.fit(
                    X_train=X_tr, y_train=y_tr,
                    X_val=X_vl, y_val=y_vl,
                )
                if t == init_size:
                    print(f"  Architecture selected: {best_hp}")

            pred = model.predict(X_test_s.values)[0]
        except Exception as e:
            print(f"  Warning: window {t}, model error: {e}")
            pred = np.nan

        results.append({
            "Quarter": Y_q.index[t],
            "Actual": Y_test.values[0],
            "Predicted": pred,
            "Window_Size": t,
        })

        print(f"  Window {t}/{N-1}: "
              f"Q={Y_q.index[t]}, "
              f"Actual={Y_test.values[0]:.4f}, "
              f"Pred={pred:.4f}")

    return pd.DataFrame(results)
 
 
def run_single(country, model_name, config, output_dir, max_lags):
    """Run one country-model combination with expanding window."""
    country_cfg = config[country]
    seed = country_cfg.get("seed", 42)
 
    print(f"\n{'='*60}")
    print(f"  {country.upper()}  |  {model_name.upper()}  |  Lags={max_lags}")
    print(f"{'='*60}")
 
    print("[1/4] Preprocessing...")
    sa_config = country_cfg.get("seasonal_adjust", None)
    data = preprocess(
        master_csv=country_cfg["master_csv"],
        target_col=country_cfg["target_col"],
        log_diff_cols=country_cfg["log_diff_cols"],
        diff_cols=country_cfg.get("diff_cols", []),
        train_ratio=country_cfg.get("train_ratio", 0.5),
        start_date=country_cfg.get("start_date"),
        end_date=country_cfg.get("end_date"),
        data_dir=os.path.join(PROJECT_ROOT, "data"),
        max_lags=max_lags,
        seasonal_adjust_cols=sa_config,
    )
    print(f"   Total quarters: {len(data.X_q_processed)}, "
          f"Features: {data.X_q_processed.shape[1]}")
 
    print("[2/4] Expanding window evaluation...")
    model_kwargs = config.get("model_defaults", {}).get(model_name, {})
 
    ew_results = expanding_window_eval(
        X_q=data.X_q_processed,
        Y_q=data.Y_q_processed,
        model_name=model_name,
        model_kwargs=model_kwargs,
        non_log_cols=data.non_log_cols,
        initial_ratio=0.6,
        seed=seed,
    )
 
    print("[3/4] Computing metrics...")
    valid = ew_results.dropna(subset=["Predicted"])
    metrics = compute_metrics(valid["Actual"].values, valid["Predicted"].values)
    if len(valid) == 0:
        print("   Warning: no valid predictions were produced; metrics are NaN.")
    else:
        print(f"   Valid evaluation windows: {len(valid)}/{len(ew_results)}")
 
    print("   Expanding Window Metrics:")
    for k, v in metrics.items():
        print(f"     {k}: {v:.4f}")
 
    print("[4/4] Saving results...")
    country_dir = os.path.join(output_dir, country)
    os.makedirs(country_dir, exist_ok=True)
    cc = COUNTRY_CODES.get(country, country[:2].upper())

    ew_results.to_csv(
        os.path.join(country_dir, f"ew_{cc}_{model_name}_lag{max_lags}.csv"), index=False
    )
    pd.DataFrame([metrics]).to_csv(
        os.path.join(country_dir, f"metrics_{cc}_{model_name}_lag{max_lags}.csv"), index=False
    )
 
    # Monthly disaggregation using the full-sample model.
    print("   Training full-sample model for disaggregation...")
    model = get_model(model_name, seed=seed, **model_kwargs)
    model.fit(
        X_train=data.X_q_train_scaled.values,
        y_train=data.Y_q_train.values,
        X_val=data.X_q_test_scaled.values,
        y_val=data.Y_q_test.values,
    )
 
    monthly_df = full_disaggregation(
        model_predict_fn=model.predict_monthly,
        X_m_scaled=data.X_m_scaled,
        X_prime_m_df=data.X_prime_m_df,
        Y_q_processed=data.Y_q_processed,
        Y_q_levels=data.Y_q_levels,
    )
    monthly_df.index.name = "DATE"
    monthly_df.to_csv(
        os.path.join(country_dir, f"monthly_gdp_{cc}_{model_name}_lag{max_lags}.csv")
    )
 
    # k_factor diagnostics
    k = monthly_df["k_factor"].dropna()
    if len(k) > 0:
        print(f"   k_factor stats: mean={k.mean():.4f}, "
              f"std={k.std():.4f}, min={k.min():.4f}, max={k.max():.4f}")

    # SHAP analysis
    try:
        from src.evaluation.shap_analysis import compute_shap_values, shap_importance_table
        print("   Computing SHAP values...")

        # Use full training data for SHAP
        X_explain = data.X_q_train_scaled.values
        feature_names = list(data.X_q_train_scaled.columns)

        # Determine model type for SHAP
        if model_name == "xgboost":
            shap_type = "tree"
            shap_model = model.model
        elif model_name == "elastic_net":
            shap_type = "linear"
            shap_model = model.model
        elif model_name == "mlp":
            shap_type = "kernel"
            shap_model = model
        else:
            shap_model = None

        if shap_model is not None:
            sv, ev = compute_shap_values(shap_model, X_explain, feature_names, shap_type)
            importance = shap_importance_table(sv, feature_names)
            importance.to_csv(
                os.path.join(country_dir, f"shap_{cc}_{model_name}_lag{max_lags}.csv"),
                index=False,
            )
            print("   SHAP importance saved.")
    except ImportError:
        print("   SHAP not installed, skipping.")
    except Exception as e:
        print(f"   SHAP failed: {e}")

    print(f"   All results saved to {country_dir}/")
    return metrics, ew_results
 
 
def main():
    parser = argparse.ArgumentParser(
        description="Monthly GDP Estimation with Expanding Window Evaluation"
    )
    parser.add_argument("--country", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--output", type=str, default="results")
    parser.add_argument("--lags", type=int, default=0,
                        help="Number of quarterly lags to add (0-3)")
    args = parser.parse_args()
 
    config = load_config(args.config)
    output_dir = os.path.join(PROJECT_ROOT, args.output)
 
    countries = COUNTRY_NAMES if args.country == "all" else [args.country]
    models = list(MODEL_REGISTRY.keys()) if args.model == "all" else [args.model]
 
    all_predictions = {}
 
    for country in countries:
        if country not in config:
            print(f"Warning: {country} not in config, skipping.")
            continue
        all_predictions[country] = {}
 
        for model_name in models:
            try:
                _, ew_df = run_single(
                    country, model_name, config, output_dir, args.lags
                )
                all_predictions[country][model_name] = ew_df
            except Exception as e:
                print(f"ERROR {country}/{model_name}: {e}")
                import traceback; traceback.print_exc()
 
    # Pairwise DM tests within each country
    for country, preds in all_predictions.items():
        model_names = list(preds.keys())
        if len(model_names) < 2:
            continue
        print(f"\n  Diebold-Mariano Tests ({country.upper()}):")
        for i in range(len(model_names)):
            for j in range(i + 1, len(model_names)):
                m1, m2 = model_names[i], model_names[j]
                df1 = preds[m1].dropna(subset=["Predicted"])
                df2 = preds[m2].dropna(subset=["Predicted"])
                # Align on common quarters
                common = df1.merge(df2, on="Quarter", suffixes=("_1", "_2"))
                if len(common) < 5:
                    continue
                dm = diebold_mariano_test(
                    common["Actual_1"].values,
                    common["Predicted_1"].values,
                    common["Predicted_2"].values,
                )
                winner = m1 if dm["model1_better"] else m2
                sig = "*" if dm["p_value"] < 0.05 else ""
                print(f"    {m1} vs {m2}: DM={dm['DM_stat']:.3f}, "
                      f"p={dm['p_value']:.3f}{sig}  (favor: {winner})")
 
 
if __name__ == "__main__":
    main()
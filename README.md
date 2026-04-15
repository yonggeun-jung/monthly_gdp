# Replication Package for Monthly GDP Disaggregation

Paper: **Temporal Disaggregation of GDP: When Does Machine Learning Help?**  
Manuscript file: `Temporal_Disaggregation_of_GDP.pdf`

## 1) Replication Scope

The replication pipeline has 5 stages:

1. Collect monthly/quarterly source series
2. Merge country-level master datasets
3. Run estimation + evaluation + temporal disaggregation
4. Compile summary result tables
5. Generate paper figures

Core entry scripts:

- `scripts/collect_data.py`
- `scripts/merge_data.py`
- `scripts/run_country.py`
- `scripts/compile_results.py`
- `scripts/generate_figures.py`

## 2) Environment Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Requirements:

- Python >= 3.9
- TensorFlow >= 2.15 (for MLP)
- Optional FRED API key: https://fred.stlouisfed.org/

## 3) Full Replication (Recommended Order)

### Step A. Collect raw data

```bash
python scripts/collect_data.py --api_key YOUR_FRED_KEY
```

This creates files under `data/raw/`.

### Step B. Build merged master files

```bash
python scripts/merge_data.py
```

This creates `data/master_China.csv`, `data/master_Germany.csv`,
`data/master_UK.csv`, and `data/master_US.csv`.

### Step C. Run country estimations

```bash
# Example: one country, all models, 2 quarterly lags
python scripts/run_country.py --country china --model all --lags 2

# Example: all countries, all models
python scripts/run_country.py --country all --model all --lags 2
```

### Step D. Compile summary tables

```bash
python scripts/compile_results.py
```

This creates:

- `results/summary_all.csv`
- `results/dm_tests_all.csv`

### Step E. Generate paper figures

```bash
python scripts/generate_figures.py
```

This creates PDF figures under `figures/`.

## 4) Output Files

For each country and model, outputs are written to `results/<country>/`:

- `ew_<CC>_<model>_lag<N>.csv` (expanding-window predictions)
- `metrics_<CC>_<model>_lag<N>.csv` (RMSE/MAE and related metrics)
- `monthly_gdp_<CC>_<model>_lag<N>.csv` (monthly disaggregated GDP)
- `shap_<CC>_<model>_lag<N>.csv` (feature importance where supported)

## 5) Manual Data Needed Before Merge

Before `scripts/merge_data.py`, place these files in `data/raw/`:

1. `uk_m1_emp.csv` (DATE, M1, Emp)
2. `de_price_compet.csv` (DATE, price_comp)
3. `Shanghai_Composite.csv` (Date, ssec)

If any are missing, merge still runs, but those indicators are excluded.

## 6) Minimal Public Repository Structure

```
config/           # Country-specific settings (train_ratio, end_date, variable lists)
scripts/
src/
figures/
requirements.txt
README.md
Temporal_Disaggregation_of_GDP.pdf
```

Local/generated artifacts (`data/raw/`, `data/master_*.csv`, `results/`,
`tuner_dir/`, caches) are intentionally excluded from version control.

## 7) Reproducibility Notes

- Use the same `--lags` value as in the paper run you want to replicate.
- Results can vary slightly for stochastic ML models unless seeds and
  hardware/software environment are fixed.
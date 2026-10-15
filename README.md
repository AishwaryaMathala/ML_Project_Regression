# Flight Price Prediction

Python scripts for the flight-price regression project: EDA, preprocessing, linear baselines, tree models, boosted models, feature engineering, tuning, blending, stacking, and hybrid approaches.

## Data layout

Place the supplied files here:

- `data/flight_prices.csv` — training file containing `price`
- `data/test.csv` — test file without `price`

The scripts expect the columns described in the report: `id`, `airline`, `flight`, `source_city`, `departure_time`, `stops`, `arrival_time`, `destination_city`, `flight_class`, `duration`, `days_left`, and `price` in training.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows, activate with `.venv\Scripts\activate`.

## Run

```bash
python eda.py
python preprocessing.py
python linear_models.py
python tree_models.py
python catboost_model.py
python boosting_models.py
python feature_engineering.py
python tune_catboost.py
python blend_models.py
python stacking_model.py
python residual_model.py
python two_stage_model.py
python direct_trees_hybrid.py
```

Outputs are written to `outputs/`. Run scripts from the repository root.

## Notes

- The report records the Direct Trees Hybrid as the highest-scoring submission (Kaggle R² 0.90327), followed by the Best Ensemble / Stacking submission (0.85522). Those are historical leaderboard values, not scores guaranteed by these scripts.
- The test file has no target labels, so its R² cannot be computed locally.
- The direct hybrid script reproduces the report's described CatBoost/XGBoost 55/45 blend. The report's highest score may depend on the exact original run, data version, split, and implementation details.
- For model selection, compare candidates on the same held-out split or cross-validation folds. The scripts print local validation R², MAE, and RMSE where applicable.
- Zero-price records and high fares are retained, consistent with the report.

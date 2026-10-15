import numpy as np
from catboost import CatBoostRegressor, CatBoostClassifier
from common import load_data, split_data, score_model, save_submission, SEED

X, y, X_test, ids = load_data()
X_train, X_val, y_train, y_val = split_data(X, y)
cat_cols = X.select_dtypes(exclude="number").columns.tolist()
for frame in (X_train, X_val, X, X_test):
    for col in cat_cols:
        frame[col] = frame[col].fillna("__MISSING__").astype(str)

threshold = 1000
gate = CatBoostClassifier(
    iterations=700, depth=5, learning_rate=0.04,
    auto_class_weights="Balanced", random_seed=SEED, verbose=False
)
gate.fit(X_train, (y_train > threshold).astype(int), cat_features=cat_cols)
base = CatBoostRegressor(
    iterations=2000, depth=7, learning_rate=0.05,
    l2_leaf_reg=5, random_seed=SEED, verbose=False
)
base.fit(X_train, y_train, cat_features=cat_cols)
special_mask = y_train > threshold
specialist = CatBoostRegressor(
    iterations=700, depth=5, learning_rate=0.04,
    l2_leaf_reg=8, random_seed=SEED, verbose=False
)
specialist.fit(X_train.loc[special_mask], y_train.loc[special_mask], cat_features=cat_cols)

p_val = gate.predict_proba(X_val)[:, 1]
base_val = base.predict(X_val)
special_val = specialist.predict(X_val)
val_pred = np.maximum(base_val + 0.5 * p_val * (special_val - base_val), 0)
score_model(y_val, val_pred, "two_stage_price_model")

gate.fit(X, (y > threshold).astype(int), cat_features=cat_cols, verbose=False)
base.fit(X, y, cat_features=cat_cols, verbose=False)
specialist.fit(X.loc[y > threshold], y.loc[y > threshold], cat_features=cat_cols, verbose=False)
p_test = gate.predict_proba(X_test)[:, 1]
test_pred = np.maximum(
    base.predict(X_test) + 0.5 * p_test * (specialist.predict(X_test) - base.predict(X_test)), 0
)
save_submission(ids, test_pred, "two_stage_submission.csv")

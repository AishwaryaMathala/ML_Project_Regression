import numpy as np
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from common import load_data, split_data, score_model, save_submission, SEED

X, y, X_test, ids = load_data()
X_train, X_val, y_train, y_val = split_data(X, y)
cat_cols = X.select_dtypes(exclude="number").columns.tolist()

for frame in (X_train, X_val, X, X_test):
    for col in cat_cols:
        frame[col] = frame[col].fillna("__MISSING__").astype(str)

cat = CatBoostRegressor(
    iterations=2453, depth=7, learning_rate=0.05, l2_leaf_reg=5,
    random_strength=0.5, random_seed=SEED, verbose=False
)
cat.fit(X_train, y_train, cat_features=cat_cols)
pred_cat_val = cat.predict(X_val)
cat.fit(X, y, cat_features=cat_cols, verbose=False)
pred_cat_test = cat.predict(X_test)

num_cols = X.select_dtypes(include="number").columns.tolist()
prep = ColumnTransformer([
    ("num", SimpleImputer(strategy="median"), num_cols),
    ("cat", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encode", OneHotEncoder(handle_unknown="ignore"))
    ]), cat_cols)
])
Xtr = prep.fit_transform(X_train)
Xva = prep.transform(X_val)
Xall = prep.fit_transform(X)
Xte = prep.transform(X_test)

lgb = LGBMRegressor(
    n_estimators=1791, learning_rate=0.02, num_leaves=31,
    reg_lambda=5, random_state=SEED, n_jobs=-1, verbosity=-1
)
lgb.fit(Xtr, y_train)
pred_lgb_val = lgb.predict(Xva)
lgb.fit(Xall, y)
pred_lgb_test = lgb.predict(Xte)

for weight in (0.60, 0.75, 0.90):
    val_pred = weight * pred_cat_val + (1 - weight) * pred_lgb_val
    test_pred = weight * pred_cat_test + (1 - weight) * pred_lgb_test
    score_model(y_val, val_pred, f"catboost_lightgbm_{weight:.2f}")
    save_submission(ids, test_pred, f"blend_catboost_{int(weight*100)}_lgb_{int((1-weight)*100)}.csv")

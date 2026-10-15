import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from common import load_data, split_data, score_model, save_submission, SEED

X, y, X_test, ids = load_data(route=True)
X_train, X_val, y_train, y_val = split_data(X, y)
cat_cols = X.select_dtypes(exclude="number").columns.tolist()
for frame in (X_train, X_val, X, X_test):
    for col in cat_cols:
        frame[col] = frame[col].fillna("__MISSING__").astype(str)

cat = CatBoostRegressor(
    iterations=3000, depth=7, learning_rate=0.04,
    l2_leaf_reg=5, random_seed=SEED, verbose=False
)
cat.fit(X_train, y_train, cat_features=cat_cols)
cat_val = cat.predict(X_val)
cat.fit(X, y, cat_features=cat_cols, verbose=False)
cat_test = cat.predict(X_test)

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

xgb = XGBRegressor(
    n_estimators=1000, learning_rate=0.03, max_depth=7,
    subsample=0.9, colsample_bytree=0.92, reg_lambda=5,
    objective="reg:squarederror", tree_method="hist",
    random_state=SEED, n_jobs=-1
)
xgb.fit(Xtr, y_train, eval_set=[(Xva, y_val)], verbose=False)
xgb_val = xgb.predict(Xva)
val_pred = np.maximum(0.55 * cat_val + 0.45 * xgb_val, 0)
score_model(y_val, val_pred, "direct_trees_hybrid")

xgb.fit(Xall, y)
test_pred = np.maximum(0.55 * cat_test + 0.45 * xgb.predict(Xte), 0)
save_submission(ids, test_pred, "submission_model3_direct_trees_hybrid.csv")

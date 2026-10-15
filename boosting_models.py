import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from common import load_data, split_data, score_model, save_submission, SEED

X, y, X_test, ids = load_data()
X_train, X_val, y_train, y_val = split_data(X, y)
num_cols = X.select_dtypes(include="number").columns.tolist()
cat_cols = X.select_dtypes(exclude="number").columns.tolist()
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
score_model(y_val, xgb.predict(Xva), "xgboost")
xgb.fit(Xall, y)
save_submission(ids, xgb.predict(Xte), "xgboost_submission.csv")

lgb = LGBMRegressor(
    n_estimators=1791, learning_rate=0.02, num_leaves=31,
    max_depth=-1, reg_lambda=5, random_state=SEED, n_jobs=-1,
    verbosity=-1
)
lgb.fit(Xtr, y_train)
score_model(y_val, lgb.predict(Xva), "lightgbm")
lgb.fit(Xall, y)
save_submission(ids, lgb.predict(Xte), "lightgbm_submission.csv")

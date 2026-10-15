import numpy as np
from catboost import CatBoostRegressor
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor, StackingRegressor
from sklearn.linear_model import Ridge
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
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
base = Pipeline([("prep", prep), ("model", ExtraTreesRegressor(
    n_estimators=400, min_samples_leaf=1, random_state=SEED, n_jobs=-1
))])
forest = Pipeline([("prep", prep), ("model", RandomForestRegressor(
    n_estimators=350, min_samples_leaf=2, random_state=SEED, n_jobs=-1
))])
model = StackingRegressor(
    estimators=[("extra", base), ("forest", forest)],
    final_estimator=Ridge(alpha=10),
    cv=5, n_jobs=-1
)
model.fit(X_train, y_train)
score_model(y_val, model.predict(X_val), "tree_stacking")
model.fit(X, y)
save_submission(ids, model.predict(X_test), "stacking_submission.csv")

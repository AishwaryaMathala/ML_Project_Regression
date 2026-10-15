import numpy as np
from catboost import CatBoostRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
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

base = CatBoostRegressor(
    iterations=2000, depth=7, learning_rate=0.05,
    l2_leaf_reg=5, random_seed=SEED, verbose=False
)
base.fit(X_train, y_train, cat_features=cat_cols)
train_base = base.predict(X_train)
val_base = base.predict(X_val)
residuals = y_train.to_numpy() - train_base

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

correction = HistGradientBoostingRegressor(
    max_iter=300, learning_rate=0.03, max_leaf_nodes=31, random_state=SEED
)
correction.fit(Xtr, residuals)
val_pred = np.maximum(val_base + correction.predict(Xva), 0)
score_model(y_val, val_pred, "residual_hist_gradient_boosting")

base.fit(X, y, cat_features=cat_cols, verbose=False)
base_train_pred = base.predict(X)
correction.fit(Xall, y.to_numpy() - base_train_pred)
test_pred = np.maximum(base.predict(X_test) + correction.predict(Xte), 0)
save_submission(ids, test_pred, "residual_hgbr_submission.csv")

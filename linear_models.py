import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from common import load_data, split_data, score_model, save_submission

X, y, X_test, ids = load_data(route=False)
X_train, X_val, y_train, y_val = split_data(X, y)
num_cols = X.select_dtypes(include="number").columns.tolist()
cat_cols = X.select_dtypes(exclude="number").columns.tolist()

def make_preprocessor():
    return ColumnTransformer([
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scale", StandardScaler())
        ]), num_cols),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encode", OneHotEncoder(handle_unknown="ignore"))
        ]), cat_cols)
    ])

models = {
    "linear_regression": LinearRegression(),
    "ridge_regression": Ridge(alpha=10),
    "lasso_regression": Lasso(alpha=1, max_iter=10000),
}

for name, estimator in models.items():
    model = Pipeline([("prep", make_preprocessor()), ("model", estimator)])
    model.fit(X_train, y_train)
    score_model(y_val, model.predict(X_val), name)
    model.fit(X, y)
    save_submission(ids, model.predict(X_test), f"{name}_submission.csv")

for degree in (2, 3):
    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("poly", PolynomialFeatures(degree=degree, include_bias=False)),
        ("scale", StandardScaler())
    ])
    prep = ColumnTransformer([
        ("num", num_pipe, num_cols),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encode", OneHotEncoder(handle_unknown="ignore"))
        ]), cat_cols)
    ])
    model = Pipeline([("prep", prep), ("model", Ridge(alpha=10))])
    model.fit(X_train, y_train)
    score_model(y_val, model.predict(X_val), f"ridge_polynomial_degree_{degree}")
    model.fit(X, y)
    save_submission(ids, model.predict(X_test), f"ridge_polynomial_degree_{degree}_submission.csv")

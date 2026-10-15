import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.model_selection import RandomizedSearchCV
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

models = {
    "decision_tree_baseline": DecisionTreeRegressor(random_state=SEED),
    "random_forest_100": RandomForestRegressor(n_estimators=100, random_state=SEED, n_jobs=-1),
    "extra_trees": ExtraTreesRegressor(n_estimators=500, min_samples_leaf=1, random_state=SEED, n_jobs=-1),
    "hist_gradient_boosting": HistGradientBoostingRegressor(max_iter=300, learning_rate=0.06, random_state=SEED)
}

for name, estimator in models.items():
    model = Pipeline([("prep", prep), ("model", estimator)])
    model.fit(X_train, y_train)
    score_model(y_val, model.predict(X_val), name)
    model.fit(X, y)
    save_submission(ids, model.predict(X_test), f"{name}_submission.csv")

tree = Pipeline([("prep", prep), ("model", DecisionTreeRegressor(random_state=SEED))])
params = {
    "model__max_depth": [6, 8, 10, 12, 16, None],
    "model__min_samples_split": [2, 5, 10, 20],
    "model__min_samples_leaf": [1, 2, 4, 8, 12],
    "model__max_features": [None, "sqrt", 0.8]
}
search = RandomizedSearchCV(tree, params, n_iter=25, scoring="r2", cv=5, n_jobs=-1, random_state=SEED)
search.fit(X_train, y_train)
print("Decision tree parameters:", search.best_params_)
score_model(y_val, search.predict(X_val), "decision_tree_tuned")
search.best_estimator_.fit(X, y)
save_submission(ids, search.best_estimator_.predict(X_test), "decision_tree_tuned_submission.csv")

forest = Pipeline([("prep", prep), ("model", RandomForestRegressor(random_state=SEED, n_jobs=-1))])
params = {
    "model__n_estimators": [300, 500, 700],
    "model__max_depth": [None, 12, 18, 24],
    "model__min_samples_leaf": [1, 2, 4],
    "model__max_features": [0.7, 0.9, 1.0]
}
search = RandomizedSearchCV(forest, params, n_iter=15, scoring="r2", cv=3, n_jobs=-1, random_state=SEED)
search.fit(X_train, y_train)
print("Random forest parameters:", search.best_params_)
score_model(y_val, search.predict(X_val), "random_forest_tuned")
search.best_estimator_.fit(X, y)
save_submission(ids, search.best_estimator_.predict(X_test), "random_forest_tuned_submission.csv")

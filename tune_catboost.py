from catboost import CatBoostRegressor
from sklearn.model_selection import RandomizedSearchCV
from common import load_data, split_data, score_model, save_submission, SEED

X, y, X_test, ids = load_data()
X_train, X_val, y_train, y_val = split_data(X, y)
cat_cols = X.select_dtypes(exclude="number").columns.tolist()
for frame in (X_train, X_val, X, X_test):
    for col in cat_cols:
        frame[col] = frame[col].fillna("__MISSING__").astype(str)

base = CatBoostRegressor(loss_function="RMSE", random_seed=SEED, verbose=False, allow_writing_files=False)
params = {
    "iterations": [1200, 1800, 2400, 3000],
    "depth": [5, 6, 7, 8, 9],
    "learning_rate": [0.02, 0.03, 0.04, 0.05, 0.07],
    "l2_leaf_reg": [3, 5, 7, 9, 12],
    "random_strength": [0.2, 0.5, 1.0]
}
search = RandomizedSearchCV(base, params, n_iter=18, cv=3, scoring="r2", n_jobs=1, random_state=SEED, verbose=1)
search.fit(X_train, y_train, cat_features=cat_cols)
print(search.best_params_)
score_model(y_val, search.best_estimator_.predict(X_val), "tuned_catboost")
best = search.best_estimator_
best.fit(X, y, cat_features=cat_cols, verbose=False)
save_submission(ids, best.predict(X_test), "tuned_catboost_submission.csv")

import pandas as pd
from catboost import CatBoostRegressor
from common import load_data, split_data, score_model, save_submission, SEED

X, y, X_test, ids = load_data()
X_train, X_val, y_train, y_val = split_data(X, y)
cat_cols = X.select_dtypes(exclude="number").columns.tolist()

for frame in (X_train, X_val, X, X_test):
    for col in cat_cols:
        frame[col] = frame[col].fillna("__MISSING__").astype(str)

model = CatBoostRegressor(
    iterations=2453,
    depth=7,
    learning_rate=0.05,
    l2_leaf_reg=5,
    random_strength=0.5,
    loss_function="RMSE",
    random_seed=SEED,
    verbose=200,
    thread_count=-1
)
model.fit(X_train, y_train, cat_features=cat_cols, eval_set=(X_val, y_val), early_stopping_rounds=200)
score_model(y_val, model.predict(X_val), "catboost")
model.fit(X, y, cat_features=cat_cols, verbose=200)
save_submission(ids, model.predict(X_test), "catboost_submission.csv")
model.save_model("outputs/catboost_model.cbm")

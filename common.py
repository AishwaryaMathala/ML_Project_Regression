from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

TRAIN_PATH = Path("data/flight_prices.csv")
TEST_PATH = Path("data/test.csv")
OUT_PATH = Path("outputs")
OUT_PATH.mkdir(parents=True, exist_ok=True)
TARGET = "price"
SEED = 42

def load_data(route=True):
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)
    y = train[TARGET].copy()
    ids = test["id"].copy() if "id" in test else pd.Series(test.index, name="id")
    X = train.drop(columns=[TARGET, "id"], errors="ignore").copy()
    X_test = test.drop(columns=["id"], errors="ignore").copy()
    if route and {"source_city", "destination_city"}.issubset(X.columns):
        for frame in (X, X_test):
            frame["route"] = frame["source_city"].fillna("missing").astype(str) + "_to_" + frame[
                "destination_city"
            ].fillna("missing").astype(str)
    for frame in (X, X_test):
        for col in frame.select_dtypes(include=["object", "category"]).columns:
            frame[col] = frame[col].fillna("__MISSING__").astype(str)
    return X, y, X_test, ids

def split_data(X, y):
    return train_test_split(X, y, test_size=0.2, random_state=SEED)

def score_model(y_true, pred, name):
    pred = np.asarray(pred)
    values = {
        "model": name,
        "r2": r2_score(y_true, pred),
        "mae": mean_absolute_error(y_true, pred),
        "rmse": mean_squared_error(y_true, pred) ** 0.5,
    }
    print(pd.Series(values).to_string())
    return values

def save_submission(ids, pred, name):
    pd.DataFrame({"id": ids, "price": np.maximum(np.asarray(pred), 0)}).to_csv(
        OUT_PATH / name, index=False
    )

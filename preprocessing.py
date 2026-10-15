from pathlib import Path
import pandas as pd

train_path = Path("data/flight_prices.csv")
test_path = Path("data/test.csv")
out_path = Path("outputs/processed")
out_path.mkdir(parents=True, exist_ok=True)

train = pd.read_csv(train_path)
test = pd.read_csv(test_path)

target = "price"
train_ids = train["id"].copy() if "id" in train else pd.Series(train.index, name="id")
test_ids = test["id"].copy() if "id" in test else pd.Series(test.index, name="id")

y = train[target].copy()
X = train.drop(columns=[target, "id"], errors="ignore").copy()
X_test = test.drop(columns=["id"], errors="ignore").copy()

for frame in (X, X_test):
    if {"source_city", "destination_city"}.issubset(frame.columns):
        frame["route"] = frame["source_city"].fillna("missing").astype(str) + "_to_" + frame[
            "destination_city"
        ].fillna("missing").astype(str)
    for col in frame.select_dtypes(include=["object", "category"]).columns:
        frame[col] = frame[col].fillna("__MISSING__").astype(str)
    for col in frame.select_dtypes(include="number").columns:
        frame[col] = frame[col].fillna(frame[col].median())

X.to_csv(out_path / "X_train.csv", index=False)
X_test.to_csv(out_path / "X_test.csv", index=False)
y.to_csv(out_path / "y_train.csv", index=False)
pd.DataFrame({"id": test_ids}).to_csv(out_path / "test_ids.csv", index=False)
print("Saved processed feature tables. Fit imputers/encoders on training folds for model evaluation.")

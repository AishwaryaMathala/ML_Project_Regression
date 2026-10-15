from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

train_path = Path("data/flight_prices.csv")
out_path = Path("outputs/eda")
out_path.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(train_path)
print("Shape:", df.shape)
print("\nColumns:\n", df.dtypes)
print("\nMissing values:\n", df.isna().sum())
print("\nDuplicates:", df.duplicated().sum())
print("\nNumeric summary:\n", df.describe(include="all").T)
print("\nTarget summary:\n", df["price"].describe())
print("Target skewness:", df["price"].skew())
print("Zero-price rows:", int((df["price"] == 0).sum()))

sns.set_theme(style="whitegrid")
fig, ax = plt.subplots(figsize=(9, 5))
sns.histplot(df["price"], bins=60, ax=ax)
ax.set_title("Flight price distribution")
fig.tight_layout()
fig.savefig(out_path / "price_distribution.png", dpi=160)
plt.close(fig)

fig, ax = plt.subplots(figsize=(9, 5))
sns.histplot(df.loc[df["price"] <= df["price"].quantile(0.99), "price"], bins=50, ax=ax)
ax.set_title("Flight prices up to the 99th percentile")
fig.tight_layout()
fig.savefig(out_path / "price_distribution_99pct.png", dpi=160)
plt.close(fig)

if "flight_class" in df:
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(data=df, x="flight_class", y="price", showfliers=False, ax=ax)
    ax.set_title("Price by flight class")
    fig.tight_layout()
    fig.savefig(out_path / "price_by_class.png", dpi=160)
    plt.close(fig)

if "days_left" in df:
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.scatterplot(data=df, x="days_left", y="price", alpha=0.35, s=18, ax=ax)
    medians = df.groupby("days_left", observed=True)["price"].median()
    ax.plot(medians.index, medians.values, linewidth=2)
    ax.set_title("Price and days before departure")
    fig.tight_layout()
    fig.savefig(out_path / "price_vs_days_left.png", dpi=160)
    plt.close(fig)

if "duration" in df:
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(df["duration"], bins=45, ax=ax)
    ax.set_title("Flight duration distribution")
    fig.tight_layout()
    fig.savefig(out_path / "duration_distribution.png", dpi=160)
    plt.close(fig)

for col in ["airline", "source_city", "destination_city", "departure_time",
            "arrival_time", "stops", "flight_class"]:
    if col in df and "price" in df:
        df.groupby(col, dropna=False)["price"].agg(["count", "mean", "median"]).sort_values(
            "median", ascending=False
        ).to_csv(out_path / f"price_by_{col}.csv")

df.isna().sum().rename("missing_count").to_csv(out_path / "missing_values.csv")
df.describe(include="all").T.to_csv(out_path / "summary.csv")
print(f"EDA files saved to {out_path}")

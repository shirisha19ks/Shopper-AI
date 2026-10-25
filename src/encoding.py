import pandas as pd
from pathlib import Path

# Project paths
ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "iqr_capped_online_shoppers.csv"
OUTPUT_PATH = ROOT / "data" / "encoded_online_shoppers.csv"

# 1. Load IQR-capped dataset
df = pd.read_csv(DATA_PATH)

print("Dataset loaded:", df.shape)
print("\nCategorical columns before encoding:")

categorical_cols = df.select_dtypes(
    include=["object", "bool", "category"]
).columns.tolist()

print(categorical_cols)

# 2. Encode target column
df["Revenue"] = df["Revenue"].map({
    True: 1,
    False: 0,
    "True": 1,
    "False": 0,
    1: 1,
    0: 0
})

if df["Revenue"].isnull().any():
    raise ValueError("Revenue encoding failed. Check target values.")

df["Revenue"] = df["Revenue"].astype(int)

# 3. Encode categorical input features
categorical_cols = [
    col for col in categorical_cols if col != "Revenue"
]

df_encoded = pd.get_dummies(
    df,
    columns=categorical_cols,
    dtype=int
)

# 4. Check encoded data
print("\nEncoded dataset shape:", df_encoded.shape)
print("\nFirst five rows:")
print(df_encoded.head())

print("\nRemaining text columns:")
print(df_encoded.select_dtypes(include=["object"]).columns.tolist())

print("\nRevenue distribution:")
print(df_encoded["Revenue"].value_counts())

# 5. Save encoded dataset
df_encoded.to_csv(OUTPUT_PATH, index=False)

print("\nEncoding completed successfully!")
print("Saved file:", OUTPUT_PATH)
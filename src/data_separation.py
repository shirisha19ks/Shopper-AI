import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

# Project path
ROOT = Path(__file__).resolve().parent.parent

# 1. Load encoded dataset
df = pd.read_csv(
    ROOT / "data" / "encoded_online_shoppers.csv"
)

print("Dataset shape:", df.shape)

# 2. Separate features (X) and target (y)
X = df.drop(columns=["Revenue"])
y = df["Revenue"].astype(int)

print("\nFeatures shape:", X.shape)
print("Target shape:", y.shape)

# 3. Display target distribution
print("\nTarget distribution:")
print(y.value_counts())

# 4. Split into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# 5. Display split shapes
print("\nTraining features:", X_train.shape)
print("Testing features:", X_test.shape)
print("Training target:", y_train.shape)
print("Testing target:", y_test.shape)

# 6. Save separated datasets
output_dir = ROOT / "data" / "splits"
output_dir.mkdir(parents=True, exist_ok=True)

X_train.to_csv(output_dir / "X_train.csv", index=False)
X_test.to_csv(output_dir / "X_test.csv", index=False)

y_train.to_csv(output_dir / "y_train.csv", index=False)
y_test.to_csv(output_dir / "y_test.csv", index=False)

print("\nData separation completed successfully!")
print("Train and test files saved in data/splits/")
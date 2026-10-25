import pandas as pd
from pathlib import Path
from imblearn.over_sampling import SMOTE

ROOT = Path(__file__).resolve().parent.parent
SPLIT_DIR = ROOT / "data" / "splits"

# Load training data only
X_train = pd.read_csv(SPLIT_DIR / "X_train.csv")
y_train = pd.read_csv(SPLIT_DIR / "y_train.csv").squeeze()

print("Before SMOTE:")
print(y_train.value_counts())

# Balance training classes
smote = SMOTE(random_state=42)

X_balanced, y_balanced = smote.fit_resample(
    X_train, y_train
)

print("\nAfter SMOTE:")
print(y_balanced.value_counts())

# Save balanced training data
X_balanced.to_csv(
    SPLIT_DIR / "X_train_balanced.csv", index=False
)

y_balanced.to_csv(
    SPLIT_DIR / "y_train_balanced.csv",
    index=False,
    header=["Revenue"]
)

print("\nBalanced training data saved successfully!")
print("Features shape:", X_balanced.shape)
print("Target shape:", y_balanced.shape)
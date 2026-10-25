import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Project paths
ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "cleaned_online_shoppers.csv"
OUTPUT_DIR = ROOT / "evaluation"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 1. Load cleaned dataset
df = pd.read_csv(DATA_PATH)

print("Dataset shape:", df.shape)
print("\nDataset loaded successfully!")

# 2. Select continuous numerical columns
numeric_cols = [
    "Administrative_Duration",
    "Informational_Duration",
    "ProductRelated_Duration",
    "BounceRates",
    "ExitRates",
    "PageValues",
    "SpecialDay"
]

numeric_cols = [col for col in numeric_cols if col in df.columns]

# 3. Box plots before outlier handling
plt.figure(figsize=(14, 7))
df[numeric_cols].boxplot(rot=45)
plt.title("Box Plot Before IQR Outlier Handling")
plt.ylabel("Values")
plt.tight_layout()

plt.savefig(OUTPUT_DIR / "boxplot_before_iqr.png")
plt.show()

# 4. Detect outliers using IQR method
outlier_report = []

for col in numeric_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)

    IQR = Q3 - Q1

    lower_limit = Q1 - 1.5 * IQR
    upper_limit = Q3 + 1.5 * IQR

    outliers = (
        (df[col] < lower_limit) |
        (df[col] > upper_limit)
    )

    outlier_report.append({
        "Column": col,
        "Q1": Q1,
        "Q3": Q3,
        "IQR": IQR,
        "Lower_Limit": lower_limit,
        "Upper_Limit": upper_limit,
        "Outlier_Count": int(outliers.sum())
    })

    print(f"\nColumn: {col}")
    print("Q1:", Q1)
    print("Q3:", Q3)
    print("IQR:", IQR)
    print("Lower limit:", lower_limit)
    print("Upper limit:", upper_limit)
    print("Outliers detected:", outliers.sum())

# 5. Save outlier report
report_df = pd.DataFrame(outlier_report)

report_df.to_csv(
    OUTPUT_DIR / "outlier_report.csv",
    index=False
)

print("\nOutlier report saved successfully!")

# 6. Encode categorical features
categorical_cols = [
    "Month",
    "OperatingSystems",
    "Browser",
    "Region",
    "TrafficType",
    "VisitorType",
    "Weekend"
]

categorical_cols = [
    col for col in categorical_cols if col in df.columns
]

# Convert target Boolean to 0 and 1
df["Revenue"] = df["Revenue"].map({
    True: 1,
    False: 0,
    "True": 1,
    "False": 0
}).astype(int)

# One-hot encoding
df_encoded = pd.get_dummies(
    df,
    columns=categorical_cols,
    dtype=int
)

# 7. Save encoded dataset
df_encoded.to_csv(
    ROOT / "data" / "encoded_online_shoppers.csv",
    index=False
)

print("\nEncoded dataset shape:", df_encoded.shape)
print("Encoding completed successfully!")
print("Encoded dataset saved in data folder.")

# 8. Display box plots for comparison
plt.figure(figsize=(14, 7))
df_encoded[numeric_cols].boxplot(rot=45)
plt.title("Box Plot After IQR Detection")
plt.ylabel("Values")
plt.tight_layout()

plt.savefig(OUTPUT_DIR / "boxplot_after_iqr.png")
plt.show()

print("\nError detection and encoding completed!")
# IQR-based outlier capping
df_iqr = df.copy()

for col in numeric_cols:
    Q1 = df_iqr[col].quantile(0.25)
    Q3 = df_iqr[col].quantile(0.75)

    IQR = Q3 - Q1

    lower_limit = Q1 - 1.5 * IQR
    upper_limit = Q3 + 1.5 * IQR

    df_iqr[col] = df_iqr[col].clip(
        lower=lower_limit,
        upper=upper_limit
    )

print("\nOutlier capping completed.")

# Save capped dataset
df_iqr.to_csv(
    ROOT / "data" / "iqr_capped_online_shoppers.csv",
    index=False
)

# Box plot after capping
plt.figure(figsize=(14, 7))
df_iqr[numeric_cols].boxplot(rot=45)
plt.title("Box Plot After IQR Capping")
plt.tight_layout()

plt.savefig(OUTPUT_DIR / "boxplot_after_iqr.png")
plt.show()

print("Capped dataset saved successfully.")
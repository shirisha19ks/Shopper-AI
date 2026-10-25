import pandas as pd
from pathlib import Path

def load_and_clean_data():

    # 1. Load dataset
    base_dir = Path(__file__).resolve().parent.parent
    file_path = base_dir / "data" / "online_shoppers_intention.csv"

    df = pd.read_csv(r"C:\Users\Dell\Desktop\Autonomous_Enterprise_Intelligence\data\online_shoppers_intention.csv")

    print("Original Dataset Shape:", df.shape)

    # 2. Check missing and null values
    print("\nMissing Values Before Cleaning:")
    print(df.isnull().sum())

    print("\nTotal Missing Values:", df.isnull().sum().sum())

    # 3. Check duplicate rows
    print("\nDuplicate Rows:", df.duplicated().sum())

    # 4. Handle missing target values
    df = df.dropna(subset=["Revenue"]).copy()

    # 5. Handle missing values in other columns
    numeric_cols = df.select_dtypes(include=["number"]).columns
    categorical_cols = df.select_dtypes(
        include=["object", "bool", "category"]
    ).columns

    # Fill numeric null values with median
    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].median())

    # Fill categorical null values with mode
    for col in categorical_cols:
        if df[col].isnull().any():
            mode_values = df[col].mode()
            if not mode_values.empty:
                df[col] = df[col].fillna(mode_values.iloc[0])

    # 6. Remove duplicate records
    df = df.drop_duplicates().copy()

    # 7. Clean text columns
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].str.strip()

    # 8. Check data types
    print("\nData Types:")
    print(df.dtypes)

    # 9. Check missing values after cleaning
    print("\nMissing Values After Cleaning:")
    print(df.isnull().sum())

    print("\nTotal Missing Values After Cleaning:",
          df.isnull().sum().sum())

    # 10. Check final dataset
    print("\nFinal Dataset Shape:", df.shape)

    print("\nFinal Duplicate Rows:", df.duplicated().sum())

    print("\nRevenue Distribution:")
    print(df["Revenue"].value_counts())

    # 11. Save cleaned dataset
    output_path = base_dir / "data" / "cleaned_online_shoppers.csv"
    df.to_csv(output_path, index=False)

    print("\nCleaned dataset saved to:", output_path)
    print("\nData preprocessing completed successfully!")

    return df


if __name__ == "__main__":
    df = load_and_clean_data()

    print("\nFirst 5 rows of cleaned dataset:")
    print(df.head())
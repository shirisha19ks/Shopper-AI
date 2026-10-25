
from pathlib import Path
import pandas as pd


# Project root directory
ROOT = Path(__file__).resolve().parent.parent

# Search for the original or cleaned dataset
DATA_PATHS = [
    ROOT / "data" / "online_shoppers_intention.csv",
    ROOT / "data" / "cleaned_data.csv",
    ROOT / "data" / "processed_data.csv",
]


def load_dataset():
    """Load the first available CSV dataset."""
    for path in DATA_PATHS:
        if path.exists():
            df = pd.read_csv(path)
            print(f"Dataset loaded: {path.name}")
            return df

    raise FileNotFoundError(
        "CSV dataset not found. Check the data folder "
        "and update DATA_PATHS with your actual CSV filename."
    )


def analyze_dataset(question="summary"):
    """Calculate real statistics from the dataset."""
    df = load_dataset()

    question = question.lower()

    if question in ["summary", "overview", "dataset summary"]:
        return {
            "rows": int(df.shape[0]),
            "columns": int(df.shape[1]),
            "column_names": df.columns.tolist(),
            "missing_values": int(df.isnull().sum().sum()),
            "duplicate_rows": int(df.duplicated().sum()),
        }

    if "revenue" in df.columns.str.lower().tolist():
        revenue_col = next(
            col for col in df.columns
            if col.lower() == "revenue"
        )

        revenue = df[revenue_col]

        if "purchase" in question or "revenue" in question:
            counts = revenue.value_counts(dropna=False)

            percentages = (
                revenue.value_counts(
                    normalize=True,
                    dropna=False
                ) * 100
            ).round(2)

            return {
                "target_column": revenue_col,
                "counts": counts.to_dict(),
                "percentages": percentages.to_dict(),
                "interpretation": (
                    "Revenue=True means a purchase occurred "
                    "during the session; False means no purchase."
                ),
            }

    if "missing" in question or "null" in question:
        return df.isnull().sum().to_dict()

    if "duplicate" in question:
        return {"duplicate_rows": int(df.duplicated().sum())}

    return {
        "message": (
            "Question not recognized. Try summary, purchase "
            "percentage, missing values, or duplicate rows."
        )
    }


if __name__ == "__main__":
    print("\nDataset Analysis Tool")
    print("Examples: summary, purchase percentage, missing values")
    print("Type exit to stop.")

    while True:
        question = input("\nEnter analysis question: ").strip()

        if question.lower() in {"exit", "quit"}:
            break

        try:
            result = analyze_dataset(question)
            print("\nAnalysis result:")
            print(result)

        except Exception as error:
            print(f"Error: {error}")

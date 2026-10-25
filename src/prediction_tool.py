
from pathlib import Path
import joblib
import pandas as pd


# Project paths
ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = ROOT / "models" / "best_tuned_model.joblib"
DATA_PATH = ROOT / "data" / "encoded_online_shoppers.csv"


def load_prediction_resources():
    """Load the trained model and encoded feature columns."""

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Trained model not found: {MODEL_PATH}"
        )

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Encoded dataset not found: {DATA_PATH}"
        )

    model = joblib.load(MODEL_PATH)
    data = pd.read_csv(DATA_PATH)

    if "Revenue" not in data.columns:
        raise ValueError(
            "Revenue target column not found in encoded dataset."
        )

    # Keep the same feature order used by the training data
    feature_columns = [
        column for column in data.columns
        if column != "Revenue"
    ]

    return model, feature_columns


def predict_purchase(session_data):
    """
    Predict purchase for one session.

    session_data must contain the encoded input feature values.
    Missing encoded features are filled with zero.
    """

    model, feature_columns = load_prediction_resources()

    # Accept a dictionary of feature names and values
    input_df = pd.DataFrame([session_data])

    # Remove target if accidentally provided
    input_df = input_df.drop(
        columns=["Revenue"],
        errors="ignore"
    )

    # Align input with the training feature columns
    input_df = input_df.reindex(
        columns=feature_columns,
        fill_value=0
    )

    input_df = input_df.apply(
        pd.to_numeric,
        errors="raise"
    )

    if hasattr(model, "predict_proba"):
        probability = float(
            model.predict_proba(input_df)[0][1]
        )
        prediction = int(probability >= 0.40)
    else:
        prediction = int(model.predict(input_df)[0])
        probability = None

    return {
        "prediction": prediction,
        "result": (
            "Purchase likely"
            if prediction == 1
            else "Purchase not likely"
        ),
        "purchase_probability": (
            round(probability * 100, 2)
            if probability is not None
            else None
        ),
        "threshold": 0.40
    }


if __name__ == "__main__":
    print("Online Shoppers Purchase Prediction Tool")

    model, features = load_prediction_resources()

    print(f"Model loaded: {MODEL_PATH.name}")
    print(f"Number of input features: {len(features)}")

    # Test using a real row from the encoded dataset.
    # Revenue is excluded from the prediction input.
    encoded_data = pd.read_csv(DATA_PATH)
    sample = encoded_data.drop(
        columns=["Revenue"]
    ).iloc[0].to_dict()

    result = predict_purchase(sample)

    print("\nTest prediction:")
    print(result)

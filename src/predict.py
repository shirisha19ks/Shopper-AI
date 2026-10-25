import pandas as pd
import joblib

from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

ROOT = Path(__file__).resolve().parent.parent

# Load best tuned model
model = joblib.load(
    ROOT / "models" / "best_tuned_model.joblib"
)

# Load test data
X_test = pd.read_csv(
    ROOT / "data" / "splits" / "X_test.csv"
)

y_test = pd.read_csv(
    ROOT / "data" / "splits" / "y_test.csv"
).squeeze().astype(int)

# Generate predictions
# Generate predictions using threshold 0.40
probabilities = model.predict_proba(X_test)[:, 1]
y_pred = (probabilities >= 0.40).astype(int)

print("\nActual class distribution:")
print(y_test.value_counts())

print("\nPredicted class distribution:")
print(pd.Series(y_pred).value_counts())

print("\nAccuracy:", round(accuracy_score(y_test, y_pred), 4))
print("Precision:", round(precision_score(y_test, y_pred, zero_division=0), 4))
print("Recall:", round(recall_score(y_test, y_pred, zero_division=0), 4))
print("F1 Score:", round(f1_score(y_test, y_pred, zero_division=0), 4))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

# Save actual and predicted results
results = pd.DataFrame({
    "Actual_Revenue": y_test.values,
    "Predicted_Revenue": y_pred
})

results.to_csv(
    ROOT / "evaluation" / "prediction_results.csv",
    index=False
)

print("\nPrediction evaluation completed.")
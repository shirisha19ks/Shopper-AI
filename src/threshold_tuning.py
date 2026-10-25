import pandas as pd
import joblib

from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

ROOT = Path(__file__).resolve().parent.parent

# Load tuned model and original test data
model = joblib.load(
    ROOT / "models" / "best_tuned_model.joblib"
)

X_test = pd.read_csv(
    ROOT / "data" / "splits" / "X_test.csv"
)

y_test = pd.read_csv(
    ROOT / "data" / "splits" / "y_test.csv"
).squeeze().astype(int)

# Predict probabilities for Purchase class
probabilities = model.predict_proba(X_test)[:, 1]
predictions = (probabilities >= 0.40).astype(int)

results = []

# Compare different thresholds
for threshold in [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]:
    predictions = (probabilities >= threshold).astype(int)

    results.append({
        "Threshold": threshold,
        "Accuracy": accuracy_score(y_test, predictions),
        "Precision": precision_score(
            y_test, predictions, zero_division=0
        ),
        "Recall": recall_score(
            y_test, predictions, zero_division=0
        ),
        "F1_Score": f1_score(
            y_test, predictions, zero_division=0
        )
    })

results_df = pd.DataFrame(results)

print("\nTHRESHOLD COMPARISON")
print(results_df.round(4).to_string(index=False))

output_path = ROOT / "evaluation" / "threshold_results.csv"
results_df.to_csv(output_path, index=False)

print("\nResults saved to:", output_path)
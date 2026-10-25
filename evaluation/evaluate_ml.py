import pandas as pd
import joblib
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report
)

ROOT = Path(__file__).resolve().parent.parent

# Load saved best model
model = joblib.load(ROOT / "models" / "best_model.joblib")

# Load original test data
X_test = pd.read_csv(ROOT / "data" / "splits" / "X_test.csv")
y_test = pd.read_csv(
    ROOT / "data" / "splits" / "y_test.csv"
).squeeze()

# Predict
y_pred = model.predict(X_test)

print("Classification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

# Plot confusion matrix
ConfusionMatrixDisplay.from_predictions(
    y_test,
    y_pred,
    display_labels=["No Purchase", "Purchase"],
    cmap="Blues"
)

plt.title("KNN Model Confusion Matrix")
plt.tight_layout()

output_path = ROOT / "evaluation" / "confusion_matrix.png"
plt.savefig(output_path)
plt.show()

print("Confusion matrix saved:", output_path)
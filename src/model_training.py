import pandas as pd
import joblib

from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

ROOT = Path(__file__).resolve().parent.parent
SPLIT_DIR = ROOT / "data" / "splits"
MODEL_DIR = ROOT / "models"
EVAL_DIR = ROOT / "evaluation"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
EVAL_DIR.mkdir(parents=True, exist_ok=True)

# 1. Load balanced training data
X_train = pd.read_csv(
    SPLIT_DIR / "X_train_balanced.csv"
)

y_train = pd.read_csv(
    SPLIT_DIR / "y_train_balanced.csv"
).squeeze()

# 2. Load original test data
X_test = pd.read_csv(
    SPLIT_DIR / "X_test.csv"
)

y_test = pd.read_csv(
    SPLIT_DIR / "y_test.csv"
).squeeze()

print("Training data shape:", X_train.shape)
print("Testing data shape:", X_test.shape)

# 3. Define four classification models
models = {
    "Logistic Regression": LogisticRegression(
        max_iter=2000,
        random_state=42
    ),

    "Decision Tree": DecisionTreeClassifier(
        random_state=42,
        class_weight="balanced"
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1
    ),

    "KNN": KNeighborsClassifier(
        n_neighbors=5,
        n_jobs=-1
    )
}

results = []
trained_models = {}

# 4. Train and evaluate each model
for name, model in models.items():

    print(f"\nTraining {name}...")

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(
        y_test, predictions, zero_division=0
    )
    recall = recall_score(
        y_test, predictions, zero_division=0
    )
    f1 = f1_score(
        y_test, predictions, zero_division=0
    )

    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1_Score": f1
    })

    trained_models[name] = model

    print(f"\n{name} Results")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")

    print("\nClassification Report:")
    print(classification_report(
        y_test, predictions, zero_division=0
    ))

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, predictions))

# 5. Compare all four models
results_df = pd.DataFrame(results)

print("\n===== FOUR MODEL COMPARISON =====")
print(results_df.to_string(index=False))

# 6. Select best model using F1 score
best_name = results_df.loc[
    results_df["F1_Score"].idxmax(), "Model"
]

best_model = trained_models[best_name]

# 7. Save best model
joblib.dump(
    best_model,
    MODEL_DIR / "best_model.joblib"
)

# 8. Save all model results
results_df.to_csv(
    EVAL_DIR / "evaluation_results.csv",
    index=False
)

print("\nBest model:", best_name)
print("Best model saved successfully!")
print("Results saved in evaluation/evaluation_results.csv")
print("\nAll four models trained successfully!")
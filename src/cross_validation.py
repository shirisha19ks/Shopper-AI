import pandas as pd
from pathlib import Path

from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline as SklearnPipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE

ROOT = Path(__file__).resolve().parent.parent
SPLIT_DIR = ROOT / "data" / "splits"
EVAL_DIR = ROOT / "evaluation"
EVAL_DIR.mkdir(exist_ok=True)

# Load original training data only
X_train = pd.read_csv(SPLIT_DIR / "X_train.csv")
y_train = pd.read_csv(SPLIT_DIR / "y_train.csv").squeeze().astype(int)

print("Training data shape:", X_train.shape)
print("\nTraining class distribution:")
print(y_train.value_counts())

# 5-fold stratified cross-validation
cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

# Four classification models
models = {
    "Logistic Regression": LogisticRegression(
        max_iter=2000,
        random_state=42
    ),
    "Decision Tree": DecisionTreeClassifier(
        random_state=42
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    ),
    "KNN": KNeighborsClassifier(
        n_neighbors=5,
        n_jobs=-1
    )
}

scoring = {
    "accuracy": "accuracy",
    "precision": "precision",
    "recall": "recall",
    "f1": "f1"
}

results = []

for name, model in models.items():
    print(f"\nRunning Cross-Validation: {name}")

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("smote", SMOTE(random_state=42)),
        ("model", model)
    ])

    scores = cross_validate(
        pipeline,
        X_train,
        y_train,
        cv=cv,
        scoring=scoring,
        n_jobs=1
    )

    results.append({
        "Model": name,
        "CV_Accuracy_Mean": scores["test_accuracy"].mean(),
        "CV_Precision_Mean": scores["test_precision"].mean(),
        "CV_Recall_Mean": scores["test_recall"].mean(),
        "CV_F1_Mean": scores["test_f1"].mean()
    })

# Display results
results_df = pd.DataFrame(results)

print("\n===== CROSS-VALIDATION RESULTS =====")
print(results_df.round(4).to_string(index=False))

# Save results
output_path = EVAL_DIR / "cross_validation_results.csv"
results_df.to_csv(output_path, index=False)

print("\nCross-validation completed successfully!")
print("Results saved to:", output_path)
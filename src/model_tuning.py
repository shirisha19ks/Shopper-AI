import pandas as pd
import joblib

from pathlib import Path
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
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
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE

ROOT = Path(__file__).resolve().parent.parent
SPLIT_DIR = ROOT / "data" / "splits"
MODEL_DIR = ROOT / "models"
EVAL_DIR = ROOT / "evaluation"

MODEL_DIR.mkdir(exist_ok=True)
EVAL_DIR.mkdir(exist_ok=True)

# 1. Load original train/test splits
X_train = pd.read_csv(SPLIT_DIR / "X_train.csv")
y_train = pd.read_csv(SPLIT_DIR / "y_train.csv").squeeze()

X_test = pd.read_csv(SPLIT_DIR / "X_test.csv")
y_test = pd.read_csv(SPLIT_DIR / "y_test.csv").squeeze()

y_train = y_train.astype(int)
y_test = y_test.astype(int)

print("Training data:", X_train.shape)
print("Testing data:", X_test.shape)

# 2. Cross-validation
cv = StratifiedKFold(
    n_splits=3,
    shuffle=True,
    random_state=42
)

# SMOTE is inside the pipeline to prevent CV data leakage
def create_pipeline(model):
    return Pipeline([
        ("scaler", StandardScaler()),
        ("smote", SMOTE(random_state=42)),
        ("model", model)
    ])

# 3. Define models and tuning parameters
search_spaces = {
    "Logistic Regression": {
        "pipeline": create_pipeline(
            LogisticRegression(max_iter=2000, random_state=42)
        ),
        "params": {
            "model__C": [0.1, 1, 10],
            "model__class_weight": [None, "balanced"]
        }
    },

    "Decision Tree": {
        "pipeline": create_pipeline(
            DecisionTreeClassifier(random_state=42)
        ),
        "params": {
            "model__max_depth": [5, 10, None],
            "model__min_samples_leaf": [2, 5],
            "model__class_weight": [None, "balanced"]
        }
    },

    "Random Forest": {
        "pipeline": create_pipeline(
            RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                n_jobs=-1
            )
        ),
        "params": {
            "model__n_estimators": [100, 200],
            "model__max_depth": [10, None],
            "model__min_samples_leaf": [2, 5]
        }
    },

    "KNN": {
        "pipeline": create_pipeline(
            KNeighborsClassifier(n_jobs=-1)
        ),
        "params": {
            "model__n_neighbors": [5, 7, 11],
            "model__weights": ["uniform", "distance"]
        }
    }
}

# 4. Fine-tune and evaluate
results = []
best_score = -1
best_name = None
best_model = None

for name, config in search_spaces.items():
    print(f"\nCross-validating and tuning: {name}")

    search = GridSearchCV(
        estimator=config["pipeline"],
        param_grid=config["params"],
        scoring="f1",
        cv=cv,
        n_jobs=-1,
        refit=True,
        error_score="raise"
    )

    search.fit(X_train, y_train)

    # Final evaluation on untouched test set
    predictions = search.best_estimator_.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(
        y_test, predictions, zero_division=0
    )
    recall = recall_score(
        y_test, predictions, zero_division=0
    )
    f1 = f1_score(y_test, predictions, zero_division=0)

    results.append({
        "Model": name,
        "CV_F1": search.best_score_,
        "Test_Accuracy": accuracy,
        "Test_Precision": precision,
        "Test_Recall": recall,
        "Test_F1": f1,
        "Best_Parameters": str(search.best_params_)
    })

    print("Best parameters:", search.best_params_)
    print(f"Mean CV F1: {search.best_score_:.4f}")
    print(f"Test accuracy: {accuracy:.4f}")
    print(f"Test precision: {precision:.4f}")
    print(f"Test recall: {recall:.4f}")
    print(f"Test F1: {f1:.4f}")

    if search.best_score_ > best_score:
        best_score = search.best_score_
        best_name = name
        best_model = search.best_estimator_

# 5. Save comparison and best tuned model
results_df = pd.DataFrame(results)
results_df.to_csv(
    EVAL_DIR / "tuning_results.csv",
    index=False
)

joblib.dump(
    best_model,
    MODEL_DIR / "best_tuned_model.joblib"
)

print("\n===== TUNING COMPARISON =====")
print(results_df[
    ["Model", "CV_F1", "Test_Accuracy", "Test_Recall", "Test_F1"]
].to_string(index=False))

print("\nBest model by cross-validation F1:", best_name)
print("Best parameters:", best_model.get_params())
print("\nSaved: models/best_tuned_model.joblib")
print("Saved: evaluation/tuning_results.csv")

# 6. Final report
final_predictions = best_model.predict(X_test)

print("\nFinal classification report:")
print(classification_report(
    y_test,
    final_predictions,
    zero_division=0
))

print("Final confusion matrix:")
print(confusion_matrix(y_test, final_predictions))

print("\nCross-validation and fine-tuning completed!")
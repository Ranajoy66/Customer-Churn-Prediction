import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline

# Load cleaned dataset
df = pd.read_csv("outputs/cleaned_churn_data.csv")

X = df.drop("Churn", axis=1)
y = df["Churn"]

# Split data before tuning
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Pipeline: SMOTE runs only on each training fold
pipeline = Pipeline([
    ("smote", SMOTE(random_state=42)),
    ("tree", DecisionTreeClassifier(random_state=42))
])

# Parameters to test
param_grid = {
    "tree__max_depth": [4, 6, 8, 10, None],
    "tree__min_samples_split": [5, 10, 20],
    "tree__min_samples_leaf": [2, 5, 10],
    "tree__criterion": ["gini", "entropy"]
}

# Tune using cross-validation
grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    cv=3,
    scoring="f1",
    n_jobs=-1,
    verbose=1
)

print("Starting hyperparameter tuning...")
grid_search.fit(X_train, y_train)

# Best model
best_model = grid_search.best_estimator_

print("\nBest parameters:")
print(grid_search.best_params_)

print("\nBest cross-validation F1-score:")
print(round(grid_search.best_score_, 4))

# Evaluate on the untouched test set
y_pred = best_model.predict(X_test)

print("\nTest Set Evaluation")
print("-------------------")
print(f"Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred, zero_division=0):.4f}")
print(f"Recall:    {recall_score(y_test, y_pred, zero_division=0):.4f}")
print(f"F1-score:  {f1_score(y_test, y_pred, zero_division=0):.4f}")

print("\nClassification report:")
print(classification_report(
    y_test,
    y_pred,
    target_names=["No Churn", "Churn"],
    zero_division=0
))

# Save tuned model and feature names
os.makedirs("models", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

joblib.dump(best_model, "models/tuned_decision_tree_smote.pkl")
joblib.dump(list(X.columns), "models/feature_names.pkl")

# Save best parameters
pd.DataFrame([grid_search.best_params_]).to_csv(
    "outputs/best_parameters.csv",
    index=False
)

print("\nTuned model saved successfully!")

import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    ConfusionMatrixDisplay,
    roc_auc_score,
    RocCurveDisplay
)

# Load dataset
df = pd.read_csv("outputs/cleaned_churn_data.csv")

X = df.drop("Churn", axis=1)
y = df["Churn"]

# Use the same split settings as Step 5
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Load tuned pipeline
model = joblib.load("models/tuned_decision_tree_smote.pkl")

# Predict
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# Evaluation metrics
print("FINAL MODEL EVALUATION")
print("----------------------")
print(f"Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred, zero_division=0):.4f}")
print(f"Recall:    {recall_score(y_test, y_pred, zero_division=0):.4f}")
print(f"F1-score:  {f1_score(y_test, y_pred, zero_division=0):.4f}")
print(f"ROC-AUC:   {roc_auc_score(y_test, y_prob):.4f}")

print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred,
    target_names=["No Churn", "Churn"],
    zero_division=0
))

os.makedirs("outputs", exist_ok=True)

# 1. Confusion matrix
ConfusionMatrixDisplay.from_predictions(
    y_test,
    y_pred,
    display_labels=["No Churn", "Churn"],
    cmap="Blues"
)
plt.title("Tuned Decision Tree - Confusion Matrix")
plt.tight_layout()
plt.savefig("outputs/final_confusion_matrix.png", dpi=150)
plt.show()

# 2. ROC curve
RocCurveDisplay.from_predictions(y_test, y_prob)
plt.title("Tuned Decision Tree - ROC Curve")
plt.tight_layout()
plt.savefig("outputs/roc_curve.png", dpi=150)
plt.show()

# 3. Feature importance
tree = model.named_steps["tree"]

importance = pd.Series(
    tree.feature_importances_,
    index=X.columns
).sort_values(ascending=False)

top_features = importance.head(15).sort_values()

plt.figure(figsize=(10, 7))
top_features.plot(kind="barh")
plt.title("Top 15 Features Influencing Churn Predictions")
plt.xlabel("Feature Importance")
plt.ylabel("Feature")
plt.tight_layout()
plt.savefig("outputs/feature_importance.png", dpi=150)
plt.show()

print("\nTop 15 Important Features:")
print(importance.head(15).round(4))

print("\nAll evaluation charts saved in the outputs folder.")

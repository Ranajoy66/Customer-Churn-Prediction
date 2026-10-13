import os
import pandas as pd

# Load dataset
file_path = "data/WA_Fn-UseC_-Telco-Customer-Churn.csv"
df = pd.read_csv(file_path)

# Remove duplicate rows
df.drop_duplicates(inplace=True)

# Remove customerID because it is an identifier, not a useful predictor
df.drop(columns=["customerID"], inplace=True, errors="ignore")

# Convert TotalCharges to numeric
df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"], errors="coerce"
)

# Handle missing values in TotalCharges
df["TotalCharges"] = df["TotalCharges"].fillna(
    df["TotalCharges"].median()
)

# Convert target column into numeric values
df["Churn"] = df["Churn"].map({
    "No": 0,
    "Yes": 1
})

# Convert categorical columns using one-hot encoding
df = pd.get_dummies(
    df,
    drop_first=True,
    dtype=int
)

# Create output directory
os.makedirs("outputs", exist_ok=True)

# Save cleaned dataset
df.to_csv("outputs/cleaned_churn_data.csv", index=False)

print("Data cleaning completed successfully!")
print("\nCleaned dataset shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())

print("\nRemaining missing values:")
print(df.isnull().sum().sum())

print("\nTarget distribution:")
print(df["Churn"].value_counts())

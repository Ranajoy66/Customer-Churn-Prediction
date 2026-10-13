
# import joblib
# import pandas as pd
# import streamlit as st

# st.set_page_config(page_title="Customer Churn Prediction")

# # Load trained model and feature names
# model = joblib.load("models/tuned_decision_tree_smote.pkl")
# feature_names = joblib.load("models/feature_names.pkl")

# df = pd.read_csv(
#     "data/WA_Fn-UseC_-Telco-Customer-Churn.csv"
# )

# st.title("Customer Churn Prediction")
# st.write("Enter customer details to predict churn probability.")

# with st.form("churn_form"):
#     gender = st.selectbox("Gender", ["Female", "Male"])
#     senior = st.selectbox("Senior Citizen", [0, 1])
#     partner = st.selectbox("Partner", ["Yes", "No"])
#     dependents = st.selectbox("Dependents", ["Yes", "No"])

#     tenure = st.number_input(
#         "Tenure (months)", min_value=0, max_value=72, value=12
#     )

#     phone = st.selectbox("Phone Service", ["Yes", "No"])
#     multiple_lines = st.selectbox(
#         "Multiple Lines", ["Yes", "No", "No phone service"]
#     )
#     internet = st.selectbox(
#         "Internet Service", ["DSL", "Fiber optic", "No"]
#     )
#     online_security = st.selectbox(
#         "Online Security", ["Yes", "No", "No internet service"]
#     )
#     online_backup = st.selectbox(
#         "Online Backup", ["Yes", "No", "No internet service"]
#     )
#     device_protection = st.selectbox(
#         "Device Protection", ["Yes", "No", "No internet service"]
#     )
#     tech_support = st.selectbox(
#         "Tech Support", ["Yes", "No", "No internet service"]
#     )
#     streaming_tv = st.selectbox(
#         "Streaming TV", ["Yes", "No", "No internet service"]
#     )
#     streaming_movies = st.selectbox(
#         "Streaming Movies", ["Yes", "No", "No internet service"]
#     )

#     contract = st.selectbox(
#         "Contract", ["Month-to-month", "One year", "Two year"]
#     )
#     paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
#     payment = st.selectbox(
#         "Payment Method",
#         [
#             "Electronic check",
#             "Mailed check",
#             "Bank transfer (automatic)",
#             "Credit card (automatic)"
#         ]
#     )

#     monthly_charges = st.number_input(
#         "Monthly Charges", min_value=0.0, value=70.0
#     )
#     total_charges = st.number_input(
#         "Total Charges", min_value=0.0, value=840.0
#     )

#     submitted = st.form_submit_button("Predict Churn")

# if submitted:
#     customer = pd.DataFrame([{
#         "gender": gender,
#         "SeniorCitizen": senior,
#         "Partner": partner,
#         "Dependents": dependents,
#         "tenure": tenure,
#         "PhoneService": phone,
#         "MultipleLines": multiple_lines,
#         "InternetService": internet,
#         "OnlineSecurity": online_security,
#         "OnlineBackup": online_backup,
#         "DeviceProtection": device_protection,
#         "TechSupport": tech_support,
#         "StreamingTV": streaming_tv,
#         "StreamingMovies": streaming_movies,
#         "Contract": contract,
#         "PaperlessBilling": paperless,
#         "PaymentMethod": payment,
#         "MonthlyCharges": monthly_charges,
#         "TotalCharges": total_charges
#     }])

#     categorical_cols = customer.select_dtypes(
#     include=["object", "str", "category"]
# ).columns.tolist()

#     # Match the encoding used during training
#     for col in categorical_cols:
#         categories = sorted(df[col].dropna().unique())
#         customer[col] = pd.Categorical(
#             customer[col], categories=categories
#         )

#     customer = pd.get_dummies(
#         customer,
#         columns=categorical_cols,
#         drop_first=True,
#         dtype=int
#     )

#     customer = customer.reindex(
#         columns=feature_names,
#         fill_value=0
#     )

#     probability = model.predict_proba(customer)[0][1] * 100

#     st.subheader("Prediction Result")
#     st.write(f"**Churn Probability: {probability:.2f}%**")
#     st.progress(float(probability / 100))

#     if probability >= 50:
#         st.warning("The model predicts a higher likelihood of churn.")
#     else:
#         st.success("The model predicts a lower likelihood of churn.")



import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Customer Churn Prediction", page_icon="📊")

# Load trained model and feature names
model = joblib.load("models/tuned_decision_tree_smote.pkl")
feature_names = joblib.load("models/feature_names.pkl")

st.title("Customer Churn Prediction")
st.write("Enter the monthly charges and tenure to predict churn.")

# User inputs
monthly_charges = st.number_input(
    "Monthly Charges (₹)",
    min_value=0.0,
    value=70.0,
    step=5.0
)

tenure = st.number_input(
    "Tenure (Months)",
    min_value=0,
    max_value=72,
    value=12
)

# Automatically calculate charges
yearly_charges = monthly_charges * 12
total_charges = monthly_charges * tenure

st.subheader("Calculated Details")
col1, col2 = st.columns(2)

with col1:
    st.metric("Yearly Charges", f"₹{yearly_charges:,.2f}")

with col2:
    st.metric("Total Charges", f"₹{total_charges:,.2f}")

# st.caption("Other customer settings use default values for prediction.")

if st.button("Predict Churn", type="primary"):

    # Default customer profile
    customer = pd.DataFrame([{
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": tenure,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "DSL",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges
    }])

    # Encode categorical columns
    categorical_cols = [
        "gender", "Partner", "Dependents", "PhoneService",
        "MultipleLines", "InternetService", "OnlineSecurity",
        "OnlineBackup", "DeviceProtection", "TechSupport",
        "StreamingTV", "StreamingMovies", "Contract",
        "PaperlessBilling", "PaymentMethod"
    ]

    for col in categorical_cols:
        customer[col] = customer[col].astype("category")

    customer = pd.get_dummies(
        customer,
        columns=categorical_cols,
        drop_first=True,
        dtype=int
    )

    # Match training feature columns
    customer = customer.reindex(
        columns=feature_names,
        fill_value=0
    )

    # Predict probability
    probabilities = model.predict_proba(customer)[0]
    churn_index = list(model.classes_).index(1)
    churn_probability = probabilities[churn_index] * 100
    no_churn_probability = 100 - churn_probability

    st.subheader("Prediction Result")

    st.metric("Churn Probability", f"{churn_probability:.2f}%")
    st.progress(float(churn_probability / 100))

    st.write(f"**Probability of staying:** {no_churn_probability:.2f}%")

    if churn_probability >= 50:
        st.warning("Higher predicted likelihood of churn.")
    else:
        st.success("Lower predicted likelihood of churn.")

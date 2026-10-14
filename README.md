# 📊 Customer Churn Prediction Using Machine Learning

## 📌 Project Overview

Customer Churn Prediction is a machine learning project designed to predict whether a customer will continue using a company's services or leave the service.

The model is trained on a historical customer dataset containing information such as monthly charges, total charges, tenure, contract type, payment method, and other relevant customer details.

By analyzing these factors, the model learns patterns associated with customers who stay with the company and customers who discontinue their subscriptions.

The main objective of this project is to help businesses identify customers who are likely to leave, understand the factors influencing churn, and develop strategies to improve customer retention.

## 🎯 Objectives

- Analyze customer subscription and billing data.
- Identify factors that influence customer churn.
- Explore the relationship between monthly charges and customer retention.
- Preprocess and prepare the dataset for machine learning.
- Train a classification model to predict customer churn.
- Evaluate the model's performance.
- Help businesses identify customers who may be at risk of leaving.

## 📂 Dataset

This project can use the Telco Customer Churn dataset, which contains customer information from a telecommunications company.

Dataset: [Telco Customer Churn on Kaggle](<https://www.kaggle.com/datasets/blastchar/telco-customer-churn>)

The dataset includes customer demographics, subscription details, service usage, billing information, and churn status.

### Key Features

- **Customer ID:** Unique identifier for each customer.
- **Tenure:** Number of months the customer has stayed with the company.
- **Monthly Charges:** Amount billed to the customer each month.
- **Total Charges:** Total amount charged to the customer over the period of service.
- **Contract:** Type of subscription contract, such as month-to-month, one-year, or two-year.
- **Payment Method:** Method used by the customer to pay bills.
- **Internet Service:** Type of internet service subscribed to by the customer.
- **Tech Support:** Indicates whether the customer has technical support services.
- **Online Security:** Indicates whether the customer has an online security subscription.
- **Churn:** Indicates whether the customer has left the company.

The exact features depend on the dataset used in the project.

### Target Variable

The target variable is **Churn**, which indicates the customer's status:

- **Yes — Customer has left the service.**
- **No — Customer continues using the service.**

The model learns from historical customer records to predict the likelihood that a customer will churn.

## ⚙️ Project Workflow

### Step 1: Data Collection

Obtain a historical customer dataset containing billing details, subscription information, service usage, and customer churn status.

The dataset should include both customers who continued using the service and customers who discontinued it.

### Step 2: Data Understanding

Examine the dataset to understand its structure, number of records, available features, and data types.

Identify the target variable and determine which customer attributes may be useful for predicting churn.

### Step 3: Data Cleaning and Preprocessing

Prepare the dataset for machine learning by handling missing values, duplicate records, and inconsistent data.

Convert categorical variables, such as contract type and payment method, into numerical representations. Scale numerical features when appropriate.

Ensure that the data used to train the model does not contain information that would only become available after the customer has already churned.

### Step 4: Exploratory Data Analysis (EDA)

Analyze customer characteristics to understand the factors associated with churn.

Important analyses include:

- Comparing monthly charges for customers who stayed and those who left.
- Studying the relationship between tenure and churn.
- Comparing churn rates across different contract types.
- Analyzing payment methods and subscription services.
- Examining total charges and the distribution of customer tenure.
- Visualizing the proportion of customers who churned versus those who remained.

This step helps identify patterns that may influence a customer's decision to leave the service.

### Step 5: Feature Selection

Select relevant customer attributes that may help predict churn.

Potentially useful features include monthly charges, total charges, tenure, contract type, payment method, and subscribed services.

The target variable, Churn, is separated from the input features so that the model learns to predict customer status from the available customer information.

### Step 6: Train-Test Split

Divide the dataset into training and testing sets.

- **Training set:** Used to teach the model patterns associated with customers who stay or leave.
- **Testing set:** Used to evaluate the model on customer records that were not included in training.

A stratified split can help preserve the proportion of churned and retained customers in both sets.

### Step 7: Model Selection and Training

Choose a suitable machine learning classification algorithm to predict customer churn.

Possible algorithms include:

- Logistic Regression
- Decision Tree
- Random Forest
- Gradient Boosting

Train the selected model using the prepared customer dataset.

The model learns relationships between customer attributes, including monthly charges and contract details, and the historical churn outcome.

### Step 8: Model Evaluation

Evaluate the trained model using the testing dataset.

Important evaluation metrics include:

- **Accuracy:** Measures the overall proportion of correct predictions.
- **Precision:** Measures how many customers predicted to churn actually churned.
- **Recall:** Measures how many actual churned customers were correctly identified.
- **F1-score:** Balances precision and recall.
- **Confusion Matrix:** Shows correct and incorrect churn predictions.
- **ROC-AUC:** Measures the model's ability to distinguish between customers who churn and customers who stay.

Accuracy should not be considered alone because the number of customers who stay may be greater than the number who leave.

### Step 9: Customer Churn Prediction

Use the trained model to predict whether a customer is likely to continue using the service or leave.

The model can analyze customer information such as:

- Monthly bill amount.
- Total charges.
- Length of time with the company.
- Contract duration.
- Payment method.
- Subscribed services.

Based on these features, the model predicts one of two outcomes:

- **No Churn:** The customer is predicted to continue using the service.
- **Churn:** The customer is predicted to leave the service.

The model may also provide a churn probability, which can help identify customers who may require additional attention.

### Step 10: Result Analysis and Business Insights

Analyze the predictions and evaluation metrics to understand how effectively the model identifies customers at risk of leaving.

The results can help businesses:

- Identify customers who may be considering leaving.
- Understand the relationship between billing amounts and churn.
- Examine whether month-to-month contracts are associated with higher churn.
- Develop customer retention campaigns.
- Offer suitable plans or support to customers at risk.
- Improve customer satisfaction and long-term retention.

Monthly charges alone do not determine whether a customer will leave. Churn predictions should consider multiple customer and service characteristics.

## 📊 Expected Outcome

The project produces a trained machine learning classification model that predicts whether a customer is likely to stay with the company or discontinue the service.

The model uses historical customer information, including billing details and subscription characteristics, to identify patterns associated with churn.

The project also provides insights into customer behavior and the factors that may influence retention.

Actual model performance depends on the dataset, selected algorithm, preprocessing methods, and model configuration.

## 🛠️ Technologies Used

- **Python:** Programming language.
- **Pandas:** Data manipulation and analysis.
- **NumPy:** Numerical computations.
- **Matplotlib:** Data visualization.
- **Seaborn:** Statistical data visualization.
- **Scikit-learn:** Machine learning model training and evaluation.
- **Jupyter Notebook:** Interactive development and data analysis.

## 🔮 Future Improvements

- Compare multiple machine learning algorithms.
- Improve prediction performance through hyperparameter tuning.
- Identify the most important factors influencing customer churn.
- Develop a dashboard to visualize customer churn patterns.
- Build a web application for interactive churn predictions.
- Deploy the trained model through an API.
- Develop customer-specific retention recommendations.
- Monitor model performance as customer behavior changes over time.

## ⚠️ Limitations

- Model performance depends on the quality and representativeness of the dataset.
- Customer behavior may change over time.
- Monthly charges alone are insufficient to explain every customer's decision to leave.
- Incorrect predictions may lead to unnecessary retention offers or missed opportunities.
- Predictions indicate potential churn risk and do not guarantee that a customer will leave.


## 👨‍💻 Author

**Ranajoy Dutta**

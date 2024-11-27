import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import streamlit as st

# Load the saved model
model = joblib.load('customer_churn_model.pkl')  # Replace with your model filename

def preprocess_data(df):
    """
    Preprocesses the given DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame to preprocess.

    Returns:
        pd.DataFrame: The preprocessed DataFrame.
    """

    # Handle missing values (if any)
    df = df.fillna(0)  # Replace with appropriate imputation strategy

    # Convert categorical columns to numerical using one-hot encoding
    categorical_cols = ['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
                        'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                        'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract',
                        'PaperlessBilling', 'PaymentMethod']
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    # Scale numerical features (if necessary)
    numerical_cols = ['tenure', 'MonthlyCharges', 'TotalCharges', 'TotalSpent', 'InvoiceNo']
    scaler = StandardScaler()
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

    return df

def predict_churn(new_data):
    """
    Preprocesses new data and predicts churn probability.

    Args:
        new_data (pd.DataFrame): A DataFrame containing new customer data.

    Returns:
        float: The predicted probability of churn for the new data.
    """

    new_data_processed = preprocess_data(new_data.copy())
    prediction_proba = model.predict_proba(new_data_processed)[:, 1]
    return prediction_proba[0]

# Streamlit app
st.title("Customer Churn Prediction App")

# Get user input for new data
gender = st.selectbox("Gender", ["Male", "Female"])
partner = st.selectbox("Partner", ["Yes", "No"])
dependents = st.selectbox("Dependents", ["Yes", "No"])
phone_service = st.selectbox("Phone Service", ["Yes", "No"])
multiple_lines = st.selectbox("Multiple Lines", ["No phone service", "No", "Yes"])
internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
online_backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
device_protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])
contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
payment_method = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
tenure = st.number_input("Tenure", min_value=0, max_value=72)
monthly_charges = st.number_input("Monthly Charges", min_value=0.0, max_value=120.0)
total_charges = st.number_input("Total Charges", min_value=0.0, max_value=9000.0)

# Create a DataFrame from user input
new_data = pd.DataFrame({
    "gender": [gender],
    "Partner": [partner],
    "Dependents": [dependents],
    "PhoneService": [phone_service],
    "MultipleLines": [multiple_lines],
    "InternetService": [internet_service],
    "OnlineSecurity": [online_security],
    "OnlineBackup": [online_backup],
    "DeviceProtection": [device_protection],
    "TechSupport": [tech_support],
    "StreamingTV": [streaming_tv],
    "StreamingMovies": [streaming_movies],
    "Contract": [contract],
    "PaperlessBilling": [paperless_billing],
    "PaymentMethod": [payment_method],
    "tenure": [tenure],
    "MonthlyCharges": [monthly_charges],
    "TotalCharges": [total_charges]
})

# Predict churn probability
if st.button("Predict Churn Probability"):
    probability = predict_churn(new_data)
    st.write(f"Predicted churn probability: {probability:.2f}")

    if probability > 0.5:
        st.warning("This customer has a high chance of churning.")
    else:
        st.success("This customer is less likely to churn.")

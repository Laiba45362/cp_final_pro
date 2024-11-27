import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline

# Load the trained pipeline
pipeline = joblib.load('customer_churn_model.pkl')

# Define a function for preprocessing new data
def preprocess_data(df):
    # Make sure to use the same columns as in training
    numerical_cols = ['tenure', 'MonthlyCharges', 'TotalCharges', 'TotalSpent']
    categorical_cols = ['gender_Female', 'Partner_Yes', 'Dependents_Yes', 
                        'PhoneService_Yes', 'MultipleLines_Yes', 'InternetService_Fiber optic', 
                        'OnlineSecurity_Yes', 'OnlineBackup_Yes', 'DeviceProtection_Yes', 
                        'TechSupport_Yes', 'StreamingTV_Yes', 'StreamingMovies_Yes', 
                        'Contract_Two year', 'PaperlessBilling_Yes', 'PaymentMethod_Credit card (automatic)']
    
    # Ensure that categorical columns are present (if any are missing, create dummy ones)
    for col in categorical_cols:
        if col not in df.columns:
            df[col] = 0  # Default to 0 for missing columns

    # Ensure numerical columns are present (fill missing with 0 or a default value)
    for col in numerical_cols:
        if col not in df.columns:
            df[col] = 0  # Default to 0 for missing numerical columns

    # Scale the numerical columns (this is important because the model was trained with scaled data)
    scaler = StandardScaler()
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

    # Return the processed DataFrame
    return df

# Function to predict churn based on the preprocessed data
def predict_churn(new_data):
    # Preprocess the new data
    new_data_processed = preprocess_data(new_data)

    # Use the pipeline to predict churn (it includes the preprocessor)
    prediction = pipeline.predict(new_data_processed)
    return prediction

# Streamlit interface
st.title("Customer Churn Prediction")

# Get user input
tenure = st.slider('Tenure (months)', 0, 72, 10)
monthly_charges = st.number_input('Monthly Charges', min_value=0.0, max_value=10000.0, value=70.0)
total_charges = st.number_input('Total Charges', min_value=0.0, max_value=100000.0, value=700.0)
total_spent = st.number_input('Total Spent', min_value=0.0, max_value=100000.0, value=700.0)

# Example categorical inputs (adjust based on your features)
gender_female = st.selectbox('Gender (Female)', [0, 1])  # Assuming binary (0 or 1)
partner_yes = st.selectbox('Partner (Yes)', [0, 1])  # Assuming binary (0 or 1)
dependents_yes = st.selectbox('Dependents (Yes)', [0, 1])  # Assuming binary (0 or 1)
phone_service_yes = st.selectbox('Phone Service (Yes)', [0, 1])  # Assuming binary (0 or 1)

# Construct the input data as a DataFrame
new_data = pd.DataFrame({
    'tenure': [tenure],
    'MonthlyCharges': [monthly_charges],
    'TotalCharges': [total_charges],
    'TotalSpent': [total_spent],
    'gender_Female': [gender_female],
    'Partner_Yes': [partner_yes],
    'Dependents_Yes': [dependents_yes],
    'PhoneService_Yes': [phone_service_yes],
    # Add other columns based on your model's features
})

# Predict churn when the user clicks the button
if st.button('Predict Churn'):
    prediction = predict_churn(new_data)
    if prediction == 1:
        st.write("The customer is likely to churn.")
    else:
        st.write("The customer is unlikely to churn.")

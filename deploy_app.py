import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline

# Load the trained model
model = joblib.load('customer_churn_model.pkl')

# Function to preprocess the new input data (similar to your original preprocessing)
def preprocess_data(input_data):
    # This should match the preprocessing steps you did before training the model
    # For simplicity, let's assume we're only preprocessing numeric features and one-hot encoding categorical features
    categorical_columns = ['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 
                          'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 
                          'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod']
    
    numerical_columns = ['tenure', 'MonthlyCharges', 'TotalCharges', 'TotalSpent']
    
    # Handle categorical columns (OneHotEncoding)
    input_data = pd.DataFrame([input_data])  # Convert input data to DataFrame if it's not already
    input_data = pd.get_dummies(input_data, columns=categorical_columns, drop_first=True)

    # Handle numerical columns (Standard scaling)
    input_data[numerical_columns] = input_data[numerical_columns].apply(pd.to_numeric, errors='coerce')
    
    # Return the preprocessed input data ready for prediction
    return input_data

# Streamlit UI
st.title("Customer Churn Prediction")

st.write("""
    This application predicts whether a customer will churn based on the features of the customer.
    Fill in the details below to predict.
""")

# User input form
with st.form(key="input_form"):
    gender = st.selectbox('Gender', ['Male', 'Female'])
    partner = st.selectbox('Partner', ['Yes', 'No'])
    dependents = st.selectbox('Dependents', ['Yes', 'No'])
    phone_service = st.selectbox('Phone Service', ['Yes', 'No'])
    internet_service = st.selectbox('Internet Service', ['DSL', 'Fiber optic', 'No'])
    online_security = st.selectbox('Online Security', ['Yes', 'No'])
    online_backup = st.selectbox('Online Backup', ['Yes', 'No'])
    device_protection = st.selectbox('Device Protection', ['Yes', 'No'])
    tech_support = st.selectbox('Tech Support', ['Yes', 'No'])
    streaming_tv = st.selectbox('Streaming TV', ['Yes', 'No'])
    streaming_movies = st.selectbox('Streaming Movies', ['Yes', 'No'])
    contract = st.selectbox('Contract', ['Month-to-month', 'One year', 'Two year'])
    paperless_billing = st.selectbox('Paperless Billing', ['Yes', 'No'])
    payment_method = st.selectbox('Payment Method', ['Electronic check', 'Mailed check', 'Bank transfer', 'Credit card'])
    
    tenure = st.number_input('Tenure (Months)', min_value=1, max_value=72, value=12)
    monthly_charges = st.number_input('Monthly Charges', min_value=0, max_value=1000, value=60)
    total_charges = st.number_input('Total Charges', min_value=0.0, max_value=100000.0, value=700.0)
    total_spent = st.number_input('Total Spending', min_value=0.0, max_value=100000.0, value=500.0)
    submit_button = st.form_submit_button(label="Predict Churn")

# Make a prediction when the user submits the form
if submit_button:
    # Prepare the input data
    input_data = {
        'gender': gender,
        'Partner': partner,
        'Dependents': dependents,
        'PhoneService': phone_service,
        'MultipleLines': 'No',  # Can be adjusted as needed
        'InternetService': internet_service,
        'OnlineSecurity': online_security,
        'OnlineBackup': online_backup,
        'DeviceProtection': device_protection,
        'TechSupport': tech_support,
        'StreamingTV': streaming_tv,
        'StreamingMovies': streaming_movies,
        'Contract': contract,
        'PaperlessBilling': paperless_billing,
        'PaymentMethod': payment_method,
        'tenure': tenure,
        'MonthlyCharges': monthly_charges,
        'TotalCharges': total_charges,
        'TotalSpent': total_spent
    }

    # Preprocess the input data
    processed_data = preprocess_data(input_data)
    
    # Make prediction
    prediction = model.predict(processed_data)
    
    # Show the result
    if prediction[0] == 'Yes':
        st.write("The customer is likely to churn.")
    else:
        st.write("The customer is not likely to churn.")

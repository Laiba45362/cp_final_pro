import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the trained model
model = joblib.load('customer_churn_model.pkl')

# Function to preprocess the data
def preprocess_data(data):
    numerical_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    categorical_cols = ['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 'InternetService', 
                        'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 
                        'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod']

    # Handle categorical columns (one-hot encoding)
    data = pd.get_dummies(data, columns=categorical_cols, drop_first=True)
    
    # Ensure the feature order matches the model's expected input features
    model_columns = ['tenure', 'MonthlyCharges', 'TotalCharges'] + [col for col in data.columns if col not in numerical_cols]
    data = data[model_columns]
    
    return data

# Define the Streamlit app
def app():
    st.title("Customer Churn Prediction")

    # Collect user input for all required features
    gender = st.selectbox("Gender", ["Male", "Female"])
    partner = st.selectbox("Partner", ["Yes", "No"])
    dependents = st.selectbox("Dependents", ["Yes", "No"])
    tenure = st.slider("Tenure (Months)", 0, 72, 12)
    monthly_charges = st.number_input("Monthly Charges", min_value=0.0, max_value=200.0, step=0.1)
    total_charges = st.number_input("Total Charges", min_value=0.0, max_value=10000.0, step=0.1)
    internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
    payment_method = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])

    # Preprocess the input data
    input_data = {
        'gender': gender,
        'Partner': partner,
        'Dependents': dependents,
        'tenure': tenure,
        'MonthlyCharges': monthly_charges,
        'TotalCharges': total_charges,
        'InternetService': internet_service,
        'Contract': contract,
        'PaperlessBilling': paperless_billing,
        'PaymentMethod': payment_method
    }

    input_df = pd.DataFrame([input_data])
    processed_input = preprocess_data(input_df)

    # Predict using the model
    prediction = model.predict(processed_input)
    prediction_proba = model.predict_proba(processed_input)[:, 1]  # Probability for class 'Yes'

    # Show the results
    if prediction == 0:
        st.write("Prediction: The customer is not likely to churn.")
    else:
        st.write("Prediction: The customer is likely to churn.")
    
    st.write(f"Probability of Churn: {prediction_proba[0]:.2f}")
    
if __name__ == "__main__":
    app()

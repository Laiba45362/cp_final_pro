import streamlit as st
import pandas as pd
import joblib

# Load the trained model
pipeline = joblib.load('customer_churn_model.pkl')

# Define a function to preprocess new data
def preprocess_data(new_data):
    # Assuming new_data is a DataFrame with the same structure as the training data
    # Perform any necessary preprocessing steps here
    # For simplicity, we'll assume the new data is already preprocessed
    return new_data

# Streamlit app
st.title('Customer Churn Prediction')

# Input fields for user to enter new data
st.header('Enter Customer Data')

# Example input fields (customize based on your dataset)
tenure = st.number_input('Tenure', min_value=0, max_value=100, value=1)
monthly_charges = st.number_input('Monthly Charges', min_value=0.0, max_value=1000.0, value=50.0)
total_charges = st.number_input('Total Charges', min_value=0.0, max_value=10000.0, value=100.0)
total_spent = st.number_input('Total Spent', min_value=0.0, max_value=10000.0, value=100.0)
invoice_count = st.number_input('Invoice Count', min_value=0, max_value=1000, value=1)

# Add more input fields as needed based on your dataset

# Create a DataFrame from the input data
new_data = pd.DataFrame({
    'tenure': [tenure],
    'MonthlyCharges': [monthly_charges],
    'TotalCharges': [total_charges],
    'TotalSpent': [total_spent],
    'InvoiceCount': [invoice_count]
    # Add more columns as needed
})

# Predict button
if st.button('Predict Churn'):
    # Preprocess the new data
    new_data_processed = preprocess_data(new_data)

    # Predict churn
    prediction = pipeline.predict(new_data_processed)

    # Display the prediction
    if prediction[0] == 1:
        st.write('The customer is likely to churn.')
    else:
        st.write('The customer is not likely to churn.')

# Run the app using the command: streamlit run app.py

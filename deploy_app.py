import streamlit as st
import pandas as pd
import joblib

# Load the preprocessing pipeline and the trained model
preprocessor_pipeline = joblib.load('churn_prediction_pipeline.pkl')
model = joblib.load('customer_churn_model.pkl')

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

# Add more input fields for categorical features
gender = st.selectbox('Gender', ['Male', 'Female'])
partner = st.selectbox('Partner', ['Yes', 'No'])
dependents = st.selectbox('Dependents', ['Yes', 'No'])
phone_service = st.selectbox('Phone Service', ['Yes', 'No'])
multiple_lines = st.selectbox('Multiple Lines', ['No phone service', 'No', 'Yes'])
internet_service = st.selectbox('Internet Service', ['DSL', 'Fiber optic', 'No'])
online_security = st.selectbox('Online Security', ['No', 'Yes', 'No internet service'])
online_backup = st.selectbox('Online Backup', ['No', 'Yes', 'No internet service'])
device_protection = st.selectbox('Device Protection', ['No', 'Yes', 'No internet service'])
tech_support = st.selectbox('Tech Support', ['No', 'Yes', 'No internet service'])
streaming_tv = st.selectbox('Streaming TV', ['No', 'Yes', 'No internet service'])
streaming_movies = st.selectbox('Streaming Movies', ['No', 'Yes', 'No internet service'])
contract = st.selectbox('Contract', ['Month-to-month', 'One year', 'Two year'])
paperless_billing = st.selectbox('Paperless Billing', ['Yes', 'No'])
payment_method = st.selectbox('Payment Method', ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'])

# Create a DataFrame from the input data
new_data = pd.DataFrame({
    'tenure': [tenure],
    'MonthlyCharges': [monthly_charges],
    'TotalCharges': [total_charges],
    'TotalSpent': [total_spent],
    'InvoiceCount': [invoice_count],
    'gender': [gender],
    'Partner': [partner],
    'Dependents': [dependents],
    'PhoneService': [phone_service],
    'MultipleLines': [multiple_lines],
    'InternetService': [internet_service],
    'OnlineSecurity': [online_security],
    'OnlineBackup': [online_backup],
    'DeviceProtection': [device_protection],
    'TechSupport': [tech_support],
    'StreamingTV': [streaming_tv],
    'StreamingMovies': [streaming_movies],
    'Contract': [contract],
    'PaperlessBilling': [paperless_billing],
    'PaymentMethod': [payment_method]
})

# Predict button
if st.button('Predict Churn'):
    # Preprocess the new data using the pipeline
    new_data_processed = preprocessor_pipeline.transform(new_data)

    # Predict churn
    prediction = model.predict(new_data_processed)

    # Display the prediction
    if prediction[0] == 1:
        st.write('The customer is likely to churn.')
    else:
        st.write('The customer is not likely to churn.')

# Run the app using the command: streamlit run app.py

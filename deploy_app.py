import streamlit as st
import pandas as pd
import joblib

# Load the trained model and scaler
model = joblib.load('customer_churn_model(1).pkl')
scaler = joblib.load('scaler.pkl')

# Function to preprocess the input data
def preprocess_data(new_data):
    # One-Hot Encoding for categorical features (same as in the original code)
    categorical_columns = ['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 
                           'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 
                           'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract', 
                           'PaperlessBilling', 'PaymentMethod']
    new_data = pd.get_dummies(new_data, columns=categorical_columns, drop_first=True)
    
    # Ensure the new data has the same columns as the training data
    missing_cols = set(X.columns) - set(new_data.columns)
    for col in missing_cols:
        new_data[col] = 0  # Add missing columns with 0 values
    new_data = new_data[X.columns]  # Ensure the columns match the training set order
    
    # Scale the new data using the scaler
    new_data_scaled = scaler.transform(new_data)
    
    return new_data_scaled

# Streamlit app
def main():
    st.title('Customer Churn Prediction')
    st.write('This app predicts whether a customer will churn based on their information.')

    # Input fields for the user to provide customer data
    customer_data = {
        'gender': st.selectbox('Gender', ['Male', 'Female']),
        'Partner': st.selectbox('Partner', ['Yes', 'No']),
        'Dependents': st.selectbox('Dependents', ['Yes', 'No']),
        'PhoneService': st.selectbox('Phone Service', ['Yes', 'No']),
        'MultipleLines': st.selectbox('Multiple Lines', ['Yes', 'No', 'No phone service']),
        'InternetService': st.selectbox('Internet Service', ['DSL', 'Fiber optic', 'No']),
        'OnlineSecurity': st.selectbox('Online Security', ['Yes', 'No', 'No internet service']),
        'OnlineBackup': st.selectbox('Online Backup', ['Yes', 'No', 'No internet service']),
        'DeviceProtection': st.selectbox('Device Protection', ['Yes', 'No', 'No internet service']),
        'TechSupport': st.selectbox('Tech Support', ['Yes', 'No', 'No internet service']),
        'StreamingTV': st.selectbox('Streaming TV', ['Yes', 'No', 'No internet service']),
        'StreamingMovies': st.selectbox('Streaming Movies', ['Yes', 'No', 'No internet service']),
        'Contract': st.selectbox('Contract', ['Month-to-month', 'One year', 'Two year']),
        'PaperlessBilling': st.selectbox('Paperless Billing', ['Yes', 'No']),
        'PaymentMethod': st.selectbox('Payment Method', ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)']),
        'tenure': st.number_input('Tenure (Months)', min_value=0, max_value=72, value=1),
        'MonthlyCharges': st.number_input('Monthly Charges', min_value=0.0, value=29.99),
        'TotalCharges': st.number_input('Total Charges', min_value=0.0, value=29.99)
    }

    # Convert the input data into a DataFrame
    new_data = pd.DataFrame(customer_data, index=[0])

    # When the user clicks the 'Predict' button
    if st.button('Predict'):
        # Preprocess the new data
        new_data_scaled = preprocess_data(new_data)

        # Make predictions using the loaded model
        prediction = model.predict(new_data_scaled)
        
        # Display the prediction result
        if prediction == 1:
            st.write("The customer is likely to churn.")
        else:
            st.write("The customer is not likely to churn.")

if __name__ == "__main__":
    main()

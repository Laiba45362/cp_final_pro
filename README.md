# cp_final_pro
https://cpfinalpro-yjhj3l75moaprpfmuzmb67.streamlit.app/


# Customer Churn Prediction - Streamlit App

This project is a customer churn prediction application built using a **Random Forest Classifier** model, which predicts whether a customer will churn (leave the service) based on various features such as contract type, internet service, monthly charges, etc.

The model is trained using a real-world dataset (`Telco Customer Churn`) and deployed via **Streamlit** for interactive predictions.

## Features:
- Predicts if a customer will churn based on input features.
- Built with **Random Forest Classifier**.
- Includes data preprocessing, feature selection, and SMOTE for handling class imbalance.
- The app allows users to input customer data and receive a churn prediction.

## Dataset

The dataset used in this project is the **Telco Customer Churn** dataset, which is available from [Kaggle](https://www.kaggle.com/blastchar/telco-customer-churn).

## Requirements

To run this app, you'll need the following Python packages:
- `pandas`
- `numpy`
- `sklearn`
- `imblearn`
- `joblib`
- `streamlit`
- `matplotlib`

You can install these dependencies using `pip`:

pip install -r requirements.txt


### requirements.txt


pandas
numpy
scikit-learn
imblearn
joblib
streamlit
matplotlib


## Steps to Train the Model

1. **Load the Dataset**: The dataset is loaded and preprocessed by handling missing values, encoding categorical variables using one-hot encoding, and scaling numerical features.
2. **Handle Class Imbalance**: SMOTE (Synthetic Minority Over-sampling Technique) is used to balance the class distribution in the target variable (`Churn`).
3. **Train-Test Split**: The data is split into training and test sets.
4. **Model Training**: A **Random Forest Classifier** is trained on the processed data.
5. **Save the Model**: The trained model and scaler are saved using `joblib` for later use in the Streamlit app.

## Steps to Run the Streamlit App

### 1. Load Pre-trained Model

The model is pre-trained and saved as `customer_churn_model.pkl`, while the scaler is saved as `scaler.pkl`. You can load these saved files in the app to make predictions on new customer data.

### 2. Run the Streamlit App

To run the Streamlit app, open a terminal and navigate to the project directory. Then run:


streamlit run app.py


This will open the Streamlit app in your browser, where you can input customer data and receive a churn prediction.

## Example Usage

1. Open the Streamlit app.
2. Input the customer data, such as `MonthlyCharges`, `Contract`, and `InternetService`.
3. The app will return a prediction of whether the customer is likely to churn or not.

## Code Overview

1. **Model Training (`train_model.py`)**:
   - This script loads and preprocesses the data, handles missing values, performs feature encoding and scaling, and trains a **Random Forest Classifier**.
   - The trained model and scaler are saved to disk using `joblib`.

2. **Streamlit App (`app.py`)**:
   - The Streamlit app loads the pre-trained model and scaler.
   - It allows users to input data and make predictions using the trained model.
   - The app also handles data preprocessing before making predictions.

### `train_model.py`

python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
import joblib

# Load dataset
telco_df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')

# Preprocess dataset
telco_df['TotalCharges'] = pd.to_numeric(telco_df['TotalCharges'], errors='coerce')
telco_df = telco_df.dropna()

categorical_columns = ['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 
                       'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 
                       'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract', 
                       'PaperlessBilling', 'PaymentMethod']

# One-hot encoding
telco_df = pd.get_dummies(telco_df, columns=categorical_columns, drop_first=True)

# Feature selection and target variable
X = telco_df.drop(columns=['Churn', 'customerID'])
y = telco_df['Churn'].map({'Yes': 1, 'No': 0})

# Handle class imbalance using SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Scaling data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train_scaled, y_train)

# Save model and scaler
joblib.dump(rf_model, 'customer_churn_model.pkl')
joblib.dump(scaler, 'scaler.pkl')


### `app.py`

python
import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load the pre-trained model and scaler
model = joblib.load('customer_churn_model.pkl')
scaler = joblib.load('scaler.pkl')

# Streamlit UI for input data
st.title("Customer Churn Prediction")
st.write("Enter customer information to predict if they will churn:")

# User input for the features
monthly_charges = st.number_input('Monthly Charges', min_value=0.0, value=100.0)
contract_type = st.selectbox('Contract Type', ['Month-to-month', 'One year', 'Two year'])
internet_service = st.selectbox('Internet Service', ['DSL', 'Fiber optic', 'None'])

# Encoding inputs
contract_mapping = {'Month-to-month': 1, 'One year': 0, 'Two year': 2}
internet_mapping = {'DSL': 1, 'Fiber optic': 2, 'None': 0}

# Prepare the input data for prediction
new_data = pd.DataFrame({
    'MonthlyCharges': [monthly_charges],
    'Contract': [contract_mapping[contract_type]],
    'InternetService': [internet_mapping[internet_service]],
})

# Scale the input data
new_data_scaled = scaler.transform(new_data)

# Make prediction
prediction = model.predict(new_data_scaled)

if prediction == 1:
    st.write("The customer is likely to churn.")
else:
    st.write("The customer is unlikely to churn.")


## Model and App Files

- **customer_churn_model.pkl`**: Pre-trained Random Forest model for churn prediction.
- **scaler.pkl`**: The StandardScaler used to scale input features.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

Feel free to fork this repository and submit pull requests with improvements or additional features. 



Enjoy using the app for customer churn prediction!


### Key Sections Explained:

1. **Introduction**: A brief description of the project, what it does, and what it predicts.
2. **Requirements**: A list of dependencies required to run the project, including how to install them via `pip`.
3. **Model Training**: A section describing the model training process (`train_model.py`), which involves data preprocessing, model training, and saving the trained model.
4. **Streamlit App**: Details on how to run the app (`app.py`), including how to input data and get predictions.
5. **Code Overview**: A breakdown of the important scripts (`train_model.py` and `app.py`).
6. **License & Contribution**: Information on licensing and how others can contribute to the project.

This structure will help others understand the purpose of your repository, how to set it up, and how to use it.

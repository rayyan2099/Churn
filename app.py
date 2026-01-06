import streamlit as st
import pandas as pd
import pickle

# --- 1. LOAD THE SAVED MODEL AND ENCODERS ---
# Use st.cache_resource for efficient loading
@st.cache_resource
def load_assets():
    """Loads the saved model and encoders from the disk."""
    try:
        with open('customer_churn_model.pkl', 'rb') as file:
            model = pickle.load(file)
        with open('encoders.pkl', 'rb') as file:
            encoders = pickle.load(file)
        return model, encoders
    except FileNotFoundError:
        st.error("Model or encoder files not found. Please ensure 'customer_churn_model.pkl' and 'encoders.pkl' are in the same directory as 'app.py'.")
        return None, None

# Load the assets
model, encoders = load_assets()

# --- 2. APP TITLE AND DESCRIPTION ---
st.set_page_config(page_title="Customer Churn Predictor", layout="wide")
st.title('Customer Churn Prediction')
st.markdown("""
This app predicts whether a customer is likely to churn based on their account information. 
Provide the customer's details in the sidebar to receive a prediction.
""")

# --- 3. SIDEBAR FOR USER INPUT ---
st.sidebar.header('Customer Details')

def get_user_input():
    """
    Creates sidebar widgets to get user input and returns a DataFrame.
    """
    # Create dropdowns (selectboxes) for categorical features and number inputs for numerical ones.
    # The options are based on the standard Telco Churn dataset.
    
    # --- Categorical Inputs ---
    gender = st.sidebar.selectbox('Gender', ('Male', 'Female'))
    partner = st.sidebar.selectbox('Partner', ('Yes', 'No'))
    dependents = st.sidebar.selectbox('Dependents', ('Yes', 'No'))
    phone_service = st.sidebar.selectbox('Phone Service', ('Yes', 'No'))
    multiple_lines = st.sidebar.selectbox('Multiple Lines', ('Yes', 'No', 'No phone service'))
    internet_service = st.sidebar.selectbox('Internet Service', ('DSL', 'Fiber optic', 'No'))
    online_security = st.sidebar.selectbox('Online Security', ('Yes', 'No', 'No internet service'))
    online_backup = st.sidebar.selectbox('Online Backup', ('Yes', 'No', 'No internet service'))
    device_protection = st.sidebar.selectbox('Device Protection', ('Yes', 'No', 'No internet service'))
    tech_support = st.sidebar.selectbox('Tech Support', ('Yes', 'No', 'No internet service'))
    streaming_tv = st.sidebar.selectbox('Streaming TV', ('Yes', 'No', 'No internet service'))
    streaming_movies = st.sidebar.selectbox('Streaming Movies', ('Yes', 'No', 'No internet service'))
    contract = st.sidebar.selectbox('Contract', ('Month-to-month', 'One year', 'Two year'))
    paperless_billing = st.sidebar.selectbox('Paperless Billing', ('Yes', 'No'))
    payment_method = st.sidebar.selectbox('Payment Method', ('Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'))

    # --- Numerical Inputs ---
    # Using sliders for a more intuitive experience
    tenure = st.sidebar.slider('Tenure (months)', 0, 72, 24)
    monthly_charges = st.sidebar.slider('Monthly Charges ($)', 18.0, 120.0, 70.0)
    total_charges = st.sidebar.slider('Total Charges ($)', 18.0, 9000.0, 1500.0)

    # --- Special Case for SeniorCitizen (0 or 1) ---
    senior_citizen_option = st.sidebar.selectbox('Senior Citizen', ('No', 'Yes'))
    senior_citizen = 1 if senior_citizen_option == 'Yes' else 0

    # Create a dictionary of the inputs
    data = {
        'gender': gender,
        'SeniorCitizen': senior_citizen,
        'Partner': partner,
        'Dependents': dependents,
        'tenure': tenure,
        'PhoneService': phone_service,
        'MultipleLines': multiple_lines,
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
        'MonthlyCharges': monthly_charges,
        'TotalCharges': total_charges
    }
    
    # Convert the dictionary to a pandas DataFrame
    # The column order MUST match the order the model was trained on.
    feature_order = ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure',
                     'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity',
                     'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
                     'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod',
                     'MonthlyCharges', 'TotalCharges']
                     
    features_df = pd.DataFrame(data, index=[0])[feature_order]
    
    return features_df

# Get user input
input_df = get_user_input()

# --- 4. PREPROCESS INPUT AND MAKE PREDICTION ---
# This step is crucial. The user input must be transformed in the same way the training data was.
# We assume 'encoders.pkl' contains a dictionary of fitted encoders.
if model and encoders:
    # Create a copy to avoid modifying the displayed user input
    processed_df = input_df.copy()
    
    # Apply the saved encoders to the categorical columns
    for column, encoder in encoders.items():
        if column in processed_df.columns:
            # The input is a DataFrame, so we transform the column
            processed_df[column] = encoder.transform(processed_df[column])

    # Display the user's selected options
    st.subheader('Your Selections')
    st.write(input_df)

    # Add a prediction button
    if st.button('Predict Churn', type="primary"):
        # Make prediction
        prediction = model.predict(processed_df)
        prediction_proba = model.predict_proba(processed_df)
        
        # The target variable 'Churn' is likely encoded as 'No'->0, 'Yes'->1
        churn_probability = prediction_proba[0][1] # Probability of the 'Churn' class

        st.subheader('Prediction Result')
        
        # Display prediction with styling
        if prediction[0] == 1:
            st.error(f'**Prediction: Customer is likely to CHURN** 😞')
        else:
            st.success(f'**Prediction: Customer is likely to STAY** 😊')

        st.metric(label="Churn Probability", value=f"{churn_probability:.2%}")
        st.progress(churn_probability)
else:
    st.warning("Please make sure the model and encoder files are available to run the app.")
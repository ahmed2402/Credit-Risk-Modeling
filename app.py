import streamlit as st
import pandas as pd
# Set pandas display options to show all columns
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)
import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler, QuantileTransformer
from sklearn.compose import ColumnTransformer

# This MUST be the first Streamlit command
st.set_page_config(page_title="Loan Prediction System", layout="wide")


# Load artifacts
@st.cache_resource
def load_artifacts():
    return {
        'xgboost': joblib.load('./models/xgboost_model.pkl'),
        'scaler': joblib.load('./models/scaler.pkl'),
        'qt' : joblib.load('./models/quantile_transformer.pkl'),
        'pycaret' : joblib.load('./models/best_model.pkl')
    }

artifacts = load_artifacts()
xgboost_model = artifacts['xgboost']
pyCaret = artifacts['pycaret']
print("Model expects features:", pyCaret.feature_names_)
# scaler = artifacts['scaler']
# qt = artifacts['qt']

def main():
    st.title("Loan Approval Prediction System")
    st.write("This application predicts whether a loan application will be approved or rejected.")

    with st.sidebar:
        st.header("Applicant Information")
        
        # Personal Information
        st.subheader("Personal Details")
        person_age = st.number_input("Age", min_value=18, max_value=100, value=30)
        person_gender = st.selectbox("Gender", ["Female", "Male"])
        person_education = st.selectbox(
            "Education Level", 
            ["High School", "Associate", "Bachelor", "Master", "Doctorate"]
        )
        
        # Financial Information
        st.subheader("Financial Details")
        person_income = st.number_input("Annual Income ($)", min_value=0, value=500000)
        person_emp_exp = st.number_input("Employment Experience (years)", min_value=0, value=5)
        credit_score = st.slider("Credit Score", 300, 850, 700)
        cb_person_cred_hist_length = st.number_input("Credit History Length (years)", min_value=0, value=5)
        previous_loan_defaults_on_file = st.selectbox("Previous Loan Defaults", ["No", "Yes"])
        
        # Loan Information
        st.subheader("Loan Details")
        loan_amnt = st.number_input("Loan Amount ($)", min_value=100, value=20000)
        loan_int_rate = st.slider("Interest Rate (%)", 0.0, 20.0, 7.5)
        loan_percent_income = st.slider("Loan Amount/Income Ratio", 0.0, 1.0, 0.3)
        loan_intent = st.selectbox(
            "Loan Purpose", 
            ["DEBTCONSOLIDATION", "EDUCATION", "HOMEIMPROVEMENT", 
             "MEDICAL", "PERSONAL", "VENTURE"]
        )
        person_home_ownership = st.selectbox(
            "Home Ownership", 
            ["MORTGAGE", "OTHER", "OWN", "RENT"]
        )
    
    # Create input dataframe with correct column order
    input_data = pd.DataFrame(index=[0], columns=[
        'person_age', 'person_gender', 'person_education', 'person_income',
        'person_emp_exp', 'loan_amnt', 'loan_int_rate', 'loan_percent_income',
        'cb_person_cred_hist_length', 'credit_score',
        'previous_loan_defaults_on_file', 'person_home_ownership_MORTGAGE',
        'person_home_ownership_OTHER', 'person_home_ownership_OWN',
        'person_home_ownership_RENT', 'loan_intent_DEBTCONSOLIDATION',
        'loan_intent_EDUCATION', 'loan_intent_HOMEIMPROVEMENT',
        'loan_intent_MEDICAL', 'loan_intent_PERSONAL', 'loan_intent_VENTURE'
    ])
    
    # Fill in the values - now they'll show up properly
    input_data['person_age'] = person_age
    input_data['person_gender'] = 1 if person_gender == "Male" else 0
    input_data['person_education'] = [
        0 if person_education == "High School" 
        else 1 if person_education == "Associate" 
        else 2 if person_education == "Bachelor" 
        else 3 if person_education == "Master" 
        else 4
    ][0]
    input_data['person_income'] = person_income
    input_data['person_emp_exp'] = person_emp_exp
    input_data['loan_amnt'] = loan_amnt
    input_data['loan_int_rate'] = loan_int_rate
    input_data['loan_percent_income'] = loan_percent_income
    input_data['cb_person_cred_hist_length'] = cb_person_cred_hist_length
    input_data['credit_score'] = credit_score
    input_data['previous_loan_defaults_on_file'] = 1 if previous_loan_defaults_on_file == "Yes" else 0
    
    # Properly handle one-hot encoded columns
    for col in ['MORTGAGE', 'OTHER', 'OWN', 'RENT']:
        input_data[f'person_home_ownership_{col}'] = 1 if person_home_ownership == col else 0
    
    for col in ['DEBTCONSOLIDATION', 'EDUCATION', 'HOMEIMPROVEMENT', 'MEDICAL', 'PERSONAL', 'VENTURE']:
        input_data[f'loan_intent_{col}'] = 1 if loan_intent == col else 0

        
    continuous_cols = ['person_income', 'person_emp_exp', 'loan_amnt', 
                 'loan_int_rate', 'loan_percent_income', 
                 'cb_person_cred_hist_length', 'credit_score']

    try:

        # At app startup, load representative training data to fit scalers
        train_data = pd.read_csv('./datasets/loan2.csv')
        train_data = train_data.drop('Unnamed: 0', axis=1)
        original_values = input_data.copy()
# Fit scalers once
        scaler = StandardScaler().fit(train_data[continuous_cols])
        qt = QuantileTransformer(output_distribution='normal').fit(train_data[['person_income']])

# Then transform new inputs using these fitted scalers
        input_data[continuous_cols] = scaler.transform(input_data[continuous_cols])
        input_data['person_income'] = qt.transform(input_data[['person_income']])
    
    # Debug output
        # print("\nOriginal Values:")
        # print(original_values.to_string())
        # print("\nTransformed Values:")
        # print(input_data[continuous_cols].to_string())
    
        if st.checkbox("Show raw input data"):
            st.write("Original Values:")
            st.dataframe(original_values)
            st.write("Transformed Values:")
            st.dataframe(input_data[continuous_cols].style.format("{:.4f}"))
        
    except Exception as e:
        st.error(f"Transformation error: {str(e)}")
        return
    
    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:  
        if st.button("Predict with XGBoost", help="Make prediction using XGBoost model"):
            try:
                prediction = xgboost_model.predict(input_data)[0]
                prediction_proba = xgboost_model.predict_proba(input_data)[0]
            
                st.subheader("XGBoost Prediction Results")
                if prediction == 1:
                    st.success("✅ Loan Approved")
                else:
                    st.error("❌ Loan Rejected")
            
                st.metric("Approval Probability", f"{prediction_proba[1]*100:.2f}%")
                st.metric("Rejection Probability", f"{prediction_proba[0]*100:.2f}%")
            
            except Exception as e:
                st.error(f"XGBoost prediction error: {str(e)}")

    with col2:
        if st.button("Predict with PyCaret", help="Make prediction using PyCaret model"):
            try:
                prediction = pyCaret.predict(input_data)[0]
                prediction_proba = pyCaret.predict_proba(input_data)[0]
            
                st.subheader("PyCaret Prediction Results")
                if prediction == 1:
                    st.success("✅ Loan Approved")
                else:
                    st.error("❌ Loan Rejected")
        
            
                st.metric("Approval Probability", f"{prediction_proba[1]*100:.2f}%")
                st.metric("Rejection Probability", f"{prediction_proba[0]*100:.2f}%")
            
            except Exception as e:
                st.error(f"PyCaret prediction error: {str(e)}")

if __name__ == "__main__":
    main()



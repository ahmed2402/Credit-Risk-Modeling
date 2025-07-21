import streamlit as st
import numpy as np
import pandas as pd
import pickle
from pathlib import Path

# Load models and transformers
MODEL_PATH = 'models/xgb_model.pkl'
SCALER_PATH = 'models/scaler.pkl'
QT_PATH = 'models/quantile_transformer.pkl'

with open(MODEL_PATH, 'rb') as f:
    model = pickle.load(f)
with open(SCALER_PATH, 'rb') as f:
    scaler = pickle.load(f)
with open(QT_PATH, 'rb') as f:
    qt = pickle.load(f)

# Feature columns as per X.csv
feature_names = [
    'person_age', 'person_gender', 'person_education', 'person_income',
    'person_emp_exp', 'loan_amnt', 'loan_int_rate', 'loan_percent_income',
    'cb_person_cred_hist_length', 'credit_score',
    'previous_loan_defaults_on_file', 'person_home_ownership_MORTGAGE',
    'person_home_ownership_OTHER', 'person_home_ownership_OWN',
    'person_home_ownership_RENT', 'loan_intent_DEBTCONSOLIDATION',
    'loan_intent_EDUCATION', 'loan_intent_HOMEIMPROVEMENT',
    'loan_intent_MEDICAL', 'loan_intent_PERSONAL', 'loan_intent_VENTURE'
]

# UI for original (raw) input
st.title('Loan Approval Prediction')

with st.sidebar:
    st.header('Enter applicant details:')
    person_age = st.number_input('Age', min_value=18, max_value=100, value=30)
    person_gender = st.selectbox('Gender', ['male', 'female'])
    person_education = st.selectbox('Education', ['High School', 'Associate', 'Bachelor', 'Master', 'Doctorate'])
    person_income = st.number_input('Annual Income', min_value=0, value=50000)
    person_emp_exp = st.number_input('Years of Employment Experience', min_value=0, max_value=80, value=5)
    person_home_ownership = st.selectbox('Home Ownership', ['RENT', 'OWN', 'MORTGAGE', 'OTHER'])
    loan_amnt = st.number_input('Loan Amount', min_value=0, value=10000)
    loan_intent = st.selectbox('Loan Intent', ['PERSONAL', 'EDUCATION', 'MEDICAL', 'VENTURE', 'HOMEIMPROVEMENT', 'DEBTCONSOLIDATION'])
    loan_int_rate = st.number_input('Interest Rate (%)', min_value=0.0, max_value=100.0, value=10.0)
    loan_percent_income = st.number_input('Loan Percent Income', min_value=0.0, max_value=1.0, value=0.2)
    cb_person_cred_hist_length = st.number_input('Credit History Length (years)', min_value=0, max_value=100, value=5)
    credit_score = st.number_input('Credit Score', min_value=0, max_value=1000, value=700)
    previous_loan_defaults_on_file = st.selectbox('Previous Loan Defaults', ['No', 'Yes'])

# Show all input values in a row
input_data = {
    'person_age': person_age,
    'person_gender': person_gender,
    'person_education': person_education,
    'person_income': person_income,
    'person_emp_exp': person_emp_exp,
    'person_home_ownership': person_home_ownership,
    'loan_amnt': loan_amnt,
    'loan_intent': loan_intent,
    'loan_int_rate': loan_int_rate,
    'loan_percent_income': loan_percent_income,
    'cb_person_cred_hist_length': cb_person_cred_hist_length,
    'credit_score': credit_score,
    'previous_loan_defaults_on_file': previous_loan_defaults_on_file
}
st.subheader('Input Values')
st.table(pd.DataFrame([input_data]))

# Checkbox for transformed data
show_transformed = st.checkbox('Show transformed data (row below)')

# Preprocessing
# 1. Encode categorical variables
# person_gender: Label Encoding (female=0, male=1)
gender_map = {'female': 0, 'male': 1}
person_gender_enc = gender_map[person_gender]

# person_education: Ordinal Encoding
education_order = ['High School', 'Associate', 'Bachelor', 'Master', 'Doctorate']
person_education_enc = education_order.index(person_education)

# previous_loan_defaults_on_file: Label Encoding (No=0, Yes=1)
defaults_map = {'No': 0, 'Yes': 1}
previous_loan_defaults_enc = defaults_map[previous_loan_defaults_on_file]

# person_home_ownership: One-hot
home_ownerships = ['MORTGAGE', 'OTHER', 'OWN', 'RENT']
home_ownership_ohe = [1 if person_home_ownership == ho else 0 for ho in home_ownerships]

# loan_intent: One-hot
loan_intents = ['DEBTCONSOLIDATION', 'EDUCATION', 'HOMEIMPROVEMENT', 'MEDICAL', 'PERSONAL', 'VENTURE']
loan_intent_ohe = [1 if loan_intent == li else 0 for li in loan_intents]

# 2. Prepare continuous features for scaling (do NOT include person_age)
continuous_features = [
    person_income, person_emp_exp, loan_amnt, loan_int_rate, loan_percent_income,
    cb_person_cred_hist_length, credit_score
]

# 3. Scale continuous features
scaled_continuous = scaler.transform(np.array([continuous_features]))[0]

# 4. Quantile transform the scaled person_income
person_income_scaled = scaled_continuous[0]
person_income_trans = qt.transform(np.array([[person_income_scaled]]))[0, 0]

# After preprocessing and before prediction
if show_transformed:
    transformed_data = {
        'person_age (raw)': person_age,
        'person_income (scaled)': scaled_continuous[0],
        'person_income (scaled + quantile)': person_income_trans,
        'person_emp_exp (scaled)': scaled_continuous[1],
        'loan_amnt (scaled)': scaled_continuous[2],
        'loan_int_rate (scaled)': scaled_continuous[3],
        'loan_percent_income (scaled)': scaled_continuous[4],
        'cb_person_cred_hist_length (scaled)': scaled_continuous[5],
        'credit_score (scaled)': scaled_continuous[6],
    }
    st.subheader('Transformed Values')
    st.table(pd.DataFrame([transformed_data]))

# 5. Assemble the final feature vector in the correct order
input_vector = [
    person_age,                  # person_age (raw, not scaled)
    person_gender_enc,           # person_gender
    person_education_enc,        # person_education
    person_income_trans,         # person_income (scaled, then quantile transformed)
    scaled_continuous[1],        # person_emp_exp
    scaled_continuous[2],        # loan_amnt
    scaled_continuous[3],        # loan_int_rate
    scaled_continuous[4],        # loan_percent_income
    scaled_continuous[5],        # cb_person_cred_hist_length
    scaled_continuous[6],        # credit_score
    previous_loan_defaults_enc,  # previous_loan_defaults_on_file
    *home_ownership_ohe,         # person_home_ownership_* (4)
    *loan_intent_ohe             # loan_intent_* (6)
]

# 6. Pad/cut to match feature_names length (should be 21)
input_vector = input_vector[:len(feature_names)]

# Prediction
if st.button('Predict Loan Approval'):
    pred = model.predict(np.array([input_vector]))[0]
    st.subheader('Prediction:')
    if pred == 1:
        st.success('Loan Approved!')
    else:
        st.error('Loan Not Approved.')

import streamlit as st
import numpy as np
import joblib

# --- PAGE SETTINGS ---
st.set_page_config(page_title="Loan Approval App", page_icon="💰", layout="centered")

# --- LOAD MODEL ---
model = joblib.load("loan_approval_model.pkl")

# --- HEADER ---
st.title("💰 Loan Approval Prediction App")
st.markdown("Enter applicant details to predict loan approval outcome.")

st.markdown("---")

# --- INPUT FIELDS ---
st.header("🔎 Applicant Information")

age = st.slider("Age", 18, 100, 30)
gender = st.radio("Gender", ["Male", "Female"])
marital = st.selectbox("Marital Status", ["Single", "Married", "Divorced", "Widowed"])
education = st.selectbox("Education Level", ["High School", "Bachelor", "Master", "PhD", "Other"])
employment = st.selectbox("Employment Status", ["Employed", "Unemployed", "Self-employed"])
income = st.number_input("Annual Income (SGD)", min_value=1000, step=1000)
loan_amt = st.number_input("Loan Amount Requested", min_value=1000, step=1000)
purpose = st.selectbox("Purpose of Loan", ["Personal", "Home", "Car", "Education", "Other"])
credit_score = st.slider("Credit Score", 300, 850, 650)
existing_loans = st.number_input("Existing Loans Count", min_value=0, step=1)
late_payments = st.number_input("Late Payments Last Year", min_value=0, step=1)

# --- ENCODING MAPS (same as training) ---
gender_map = {"Male": 1, "Female": 0}
marital_map = {"Single": 0, "Married": 1, "Divorced": 2, "Widowed": 3}
education_map = {"High School": 0, "Bachelor": 1, "Master": 2, "PhD": 3, "Other": 4}
employment_map = {"Employed": 0, "Unemployed": 1, "Self-employed": 2}
purpose_map = {"Personal": 0, "Home": 1, "Car": 2, "Education": 3, "Other": 4}

# --- FEATURE ENGINEERING ---
debt_to_income = loan_amt / income if income != 0 else 0
loan_burden = loan_amt / (existing_loans + 1)
late_payment_ratio = late_payments / (existing_loans + 1)

# --- FINAL INPUT ARRAY (14 features) ---
input_data = np.array([[ 
    age,
    gender_map[gender],
    marital_map[marital],
    education_map[education],
    employment_map[employment],
    income,
    loan_amt,
    purpose_map[purpose],
    credit_score,
    existing_loans,
    late_payments,
    debt_to_income,
    loan_burden,
    late_payment_ratio
]])

# --- PREDICT ---
st.markdown("### 📊 Prediction")
if st.button("Check Loan Approval"):
    prediction = model.predict(input_data)[0]
    if prediction == 1:
        st.success("✅ Loan is **likely to be Approved**.")
    else:
        st.error("❌ Loan is **likely to be Rejected**.")

# --- FOOTER ---
st.markdown("---")
st.caption("Created by [Your Name] • Specialist Diploma in AI Solutions • Temasek Polytechnic")

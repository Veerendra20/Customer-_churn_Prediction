import streamlit as st
import joblib
import pandas as pd

model = joblib.load('churn_model.pkl')

st.title("Customer Churn Prediction")

gender = st.selectbox("Gender", ["Male", "Female"])
SeniorCitizen = st.selectbox("Senior Citizen", [0, 1])
Partner = st.selectbox("Partner", ["Yes", "No"])
Dependents = st.selectbox("Dependents", ["Yes", "No"])
tenure = st.number_input("Tenure", min_value=0)
PhoneService = st.selectbox("Phone Service", ["Yes", "No"])
MonthlyCharges = st.number_input("Monthly Charges", min_value=0.0)
TotalCharges = st.number_input("Total Charges", min_value=0.0)

if st.button("Predict"):
    input_data = {
        'gender': gender,
        'SeniorCitizen': SeniorCitizen,
        'Partner': Partner,
        'Dependents': Dependents,
        'tenure': tenure,
        'PhoneService': PhoneService,
        'MonthlyCharges': MonthlyCharges,
        'TotalCharges': TotalCharges
    }

    input_df = pd.DataFrame([input_data])
    input_df = pd.get_dummies(input_df)
    input_df = input_df.reindex(columns=model.feature_names_in_, fill_value=0)

    prediction = model.predict(input_df)[0]

    if prediction == 1:
        st.error("🔴 This customer is likely to CHURN.")
    else:
        st.success("🟢 This customer is likely to STAY.")
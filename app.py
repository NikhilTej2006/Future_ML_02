# app.py (Enhanced Streamlit UI with Logging)
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import datetime

# Load Model and Columns
model = pickle.load(open("model/xgb_churn_model.pkl", "rb"))
with open("model/feature_columns.pkl", "rb") as f:
    input_columns = pickle.load(f)

# App Configuration
st.set_page_config(page_title="Customer Churn Predictor", layout="centered")

# Custom CSS Styling
st.markdown("""
    <style>
        body {
            background-color: #f4f4f4;
        }
        .main-title {
            font-size: 42px;
            font-weight: bold;
            text-align: center;
            color: #003366;
            margin-bottom: 10px;
        }
        .sub-text {
            text-align: center;
            color: #777;
            margin-bottom: 20px;
        }
        .stButton>button {
            background-color: #003366;
            color: white;
            border-radius: 12px;
            padding: 10px 24px;
            font-size: 16px;
        }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown("<div class='main-title'>📊 Customer Churn Prediction</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-text'>Enter customer details to predict churn likelihood.</div>", unsafe_allow_html=True)

# Sidebar
st.sidebar.header("📝 Input Customer Details")

# Input Function
def user_input():
    gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
    senior = st.sidebar.selectbox("Senior Citizen", [0, 1])
    partner = st.sidebar.selectbox("Partner", ["Yes", "No"])
    dependents = st.sidebar.selectbox("Dependents", ["Yes", "No"])
    tenure = st.sidebar.slider("Tenure (months)", 0, 72, 12)
    phone_service = st.sidebar.selectbox("Phone Service", ["Yes", "No"])
    multiple_lines = st.sidebar.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
    internet_service = st.sidebar.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    contract = st.sidebar.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    monthly_charges = st.sidebar.slider("Monthly Charges", 0.0, 150.0, 70.0)

    data = {
        "gender": gender,
        "SeniorCitizen": senior,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "Contract": contract,
        "MonthlyCharges": monthly_charges,
    }
    return pd.DataFrame([data])

input_df = user_input()

# Prepare input for model
input_df_enc = pd.get_dummies(input_df)
input_df_enc = input_df_enc.reindex(columns=input_columns, fill_value=0)

# Prediction
if st.button("🔍 Predict Churn"):
    churn_prob = model.predict_proba(input_df_enc)[0][1]
    churn_pred = model.predict(input_df_enc)[0]

    st.markdown("---")
    st.subheader("📢 Prediction Results")
    st.metric("Churn Probability", f"{churn_prob * 100:.2f}%")

    if churn_pred == 1:
        st.error("⚠️ This customer is likely to churn. Take retention action.")
    else:
        st.success("✅ This customer is likely to stay. Keep up the good work!")

    # Show input summary
    st.markdown("### 🔎 Customer Input Summary")
    st.dataframe(input_df, use_container_width=True)

    # Log predictions to CSV for Power BI
    log_df = input_df.copy()
    log_df["Churn_Probability"] = churn_prob
    log_df["Churn_Predicted"] = churn_pred
    log_df["Timestamp"] = datetime.datetime.now()

    if not os.path.exists("powerbi_predictions.csv"):
        log_df.to_csv("powerbi_predictions.csv", index=False)
    else:
        log_df.to_csv("powerbi_predictions.csv", mode='a', header=False, index=False)

# Footer
st.markdown("""
    <br><hr>
    <div style='text-align:center; color: #888;'>Made with Future Interns</div>
""", unsafe_allow_html=True)

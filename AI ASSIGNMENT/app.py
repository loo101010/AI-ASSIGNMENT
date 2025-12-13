import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler

# =============================
# Load saved artifacts
# =============================
@st.cache_resource
def load_artifacts():
    with open('scaler.joblib', 'rb') as f:
        scaler = pickle.load(f)
    with open('feature_names.pkl', 'rb') as f:
        feature_names = pickle.load(f)
    with open('ann_model.keras', 'rb') as f:
        ann_model = pickle.load(f)
    with open('svm_model.joblib', 'rb') as f:
        svm_model = pickle.load(f)
    with open('knn_model.joblib', 'rb') as f:
        knn_model = pickle.load(f)
    return scaler, feature_names, ann_model, svm_model, knn_model

scaler, feature_names, ann_model, svm_model, knn_model = load_artifacts()

# =============================
# Streamlit UI
# =============================
st.set_page_config(page_title="Customer Churn Prediction", layout="wide")

st.title("📊 Customer Churn Prediction App")
st.markdown("Predict whether a customer is likely to **churn** using trained ML models.")

# Sidebar model selection
st.sidebar.header("⚙️ Model Selection")
model_choice = st.sidebar.selectbox(
    "Choose Model",
    ("ANN", "SVM", "KNN")
)

# =============================
# Input Form
# =============================
st.header("🧾 Customer Information")

col1, col2, col3 = st.columns(3)

with col1:
    gender = st.selectbox("Gender", ["Male", "Female"])
    SeniorCitizen = st.selectbox("Senior Citizen", [0, 1])
    Partner = st.selectbox("Partner", ["Yes", "No"])
    Dependents = st.selectbox("Dependents", ["Yes", "No"])

with col2:
    tenure = st.slider("Tenure (months)", 0, 72, 12)
    PhoneService = st.selectbox("Phone Service", ["Yes", "No"])
    PaperlessBilling = st.selectbox("Paperless Billing", ["Yes", "No"])
    MonthlyCharges = st.number_input("Monthly Charges", min_value=0.0, value=70.0)

with col3:
    TotalCharges = st.number_input("Total Charges", min_value=0.0, value=1000.0)
    Contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    PaymentMethod = st.selectbox(
        "Payment Method",
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)",
        ],
    )

# =============================
# Build input dataframe
# =============================
def build_input():
    data = {
        'gender': 1 if gender == 'Male' else 0,
        'SeniorCitizen': SeniorCitizen,
        'Partner': 1 if Partner == 'Yes' else 0,
        'Dependents': 1 if Dependents == 'Yes' else 0,
        'tenure': tenure,
        'PhoneService': 1 if PhoneService == 'Yes' else 0,
        'PaperlessBilling': 1 if PaperlessBilling == 'Yes' else 0,
        'MonthlyCharges': MonthlyCharges,
        'TotalCharges': TotalCharges,
    }

    df_input = pd.DataFrame([data])

    # One-hot placeholders
    for col in feature_names:
        if col not in df_input.columns:
            df_input[col] = 0

    # Contract encoding
    if Contract != 'Month-to-month':
        df_input[f'Contract_{Contract}'] = 1

    # Payment method encoding
    df_input[f'PaymentMethod_{PaymentMethod}'] = 1

    df_input = df_input[feature_names]
    return df_input

# =============================
# Prediction
# =============================
st.markdown("---")

if st.button("🔍 Predict Churn"):
    input_df = build_input()
    input_scaled = scaler.transform(input_df)

    if model_choice == "ANN":
        model = ann_model
    elif model_choice == "SVM":
        model = svm_model
    else:
        model = knn_model

    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]

    st.subheader("📈 Prediction Result")

    if prediction == 1:
        st.error(f"⚠️ Customer is likely to churn (Probability: {probability:.2%})")
    else:
        st.success(f"✅ Customer is not likely to churn (Probability: {probability:.2%})")

# =============================
# Footer
# =============================
st.markdown("---")
st.markdown("✅ Models used: ANN, SVM, KNN")
st.markdown("🚀 Built with Streamlit")

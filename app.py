import streamlit as st
import pandas as pd
import numpy as np
import pickle

# ===============================
# Page Configuration
# ===============================
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)

# ===============================
# Load Trained Models and Scaler
# ===============================
@st.cache_resource
def load_resources():
    with open("ann_model.keras", "rb") as f:
        ann_model = pickle.load(f)

    with open("svm_model.joblib", "rb") as f:
        svm_model = pickle.load(f)

    with open("knn_model.joblib", "rb") as f:
        knn_model = pickle.load(f)

    with open("scaler.joblib", "rb") as f:
        scaler = pickle.load(f)

    with open("feature_names.pkl", "rb") as f:
        feature_names = pickle.load(f)

    return ann_model, svm_model, knn_model, scaler, feature_names


ann_model, svm_model, knn_model, scaler, feature_names = load_resources()

# ===============================
# Sidebar - Model Selection
# ===============================
st.sidebar.title("⚙️ Model Settings")

model_choice = st.sidebar.selectbox(
    "Select Model",
    ["ANN", "SVM", "KNN"]
)

st.sidebar.markdown("---")
st.sidebar.info(
    "This application predicts whether a customer is likely "
    "to churn based on their service information."
)

# ===============================
# Main Title
# ===============================
st.title("📊 Customer Churn Prediction System")
st.markdown(
    """
    Input your data
    """
)

# ===============================
# Customer Input Form
# ===============================
st.header("🧾 Customer Information")

col1, col2, col3 = st.columns(3)

with col1:
    gender = st.selectbox("Gender", ["Male", "Female"])
    senior = st.selectbox("Senior Citizen", [0, 1])
    partner = st.selectbox("Partner", ["Yes", "No"])
    dependents = st.selectbox("Dependents", ["Yes", "No"])
    tenure = st.slider("Tenure (months)", 0, 72, 12)

with col2:
    phone_service = st.selectbox("Phone Service", ["Yes", "No"])
    paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
    monthly_charges = st.number_input("Monthly Charges", 0.0, 200.0, 70.0)
    total_charges = st.number_input("Total Charges", 0.0, 10000.0, 1000.0)

with col3:
    contract = st.selectbox(
        "Contract Type",
        ["Month-to-month", "One year", "Two year"]
    )

    internet = st.selectbox(
        "Internet Service",
        ["DSL", "Fiber optic", "No"]
    )

    payment = st.selectbox(
        "Payment Method",
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)"
        ]
    )

# ===============================
# Prepare Input Data
# ===============================
def prepare_input_data():
    data = {
        "gender": 1 if gender == "Male" else 0,
        "SeniorCitizen": senior,
        "Partner": 1 if partner == "Yes" else 0,
        "Dependents": 1 if dependents == "Yes" else 0,
        "tenure": tenure,
        "PhoneService": 1 if phone_service == "Yes" else 0,
        "PaperlessBilling": 1 if paperless == "Yes" else 0,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
    }

    input_df = pd.DataFrame([data])

    # Initialize all one-hot encoded columns as 0
    for col in feature_names:
        if col not in input_df.columns:
            input_df[col] = 0

    # Contract
    if contract != "Month-to-month":
        input_df[f"Contract_{contract}"] = 1

    # Internet Service
    if internet != "DSL":
        input_df[f"InternetService_{internet}"] = 1

    # Payment Method
    input_df[f"PaymentMethod_{payment}"] = 1

    # Reorder columns to match training data
    input_df = input_df[feature_names]

    return input_df

# ===============================
# Prediction Section
# ===============================
st.markdown("---")
st.header("🔮 Prediction Result")

if st.button("Predict Churn"):

    input_df = prepare_input_data()
    input_scaled = scaler.transform(input_df)

    if model_choice == "ANN":
        model = ann_model
    elif model_choice == "SVM":
        model = svm_model
    else:
        model = knn_model

    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]

    if prediction == 1:
        st.error(
            f"⚠️ Customer is **likely to churn**\n\n"
            f"Churn Probability: **{probability:.2%}**"
        )
    else:
        st.success(
            f"✅ Customer is **not likely to churn**\n\n"
            f"Churn Probability: **{probability:.2%}**"
        )

# ===============================
# Footer
# ===============================
st.markdown("---")
st.caption(
    "Customer Churn Prediction System | ANN, SVM, KNN | Streamlit Deployment"
)


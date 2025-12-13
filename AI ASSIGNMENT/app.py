import streamlit as st
import pandas as pd
import numpy as np
import pickle

from sklearn.preprocessing import StandardScaler

# ===============================
# Page Configuration
# ===============================
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)

# ===============================
# Load Models and Scaler
# ===============================
@st.cache_resource
def load_models():
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


ann_model, svm_model, knn_model, scaler, feature_names = load_models()

# ===============================
# Sidebar - Model Selection
# ===============================
st.sidebar.title("⚙️ Settings")

model_choice = st.sidebar.selectbox(
    "Select Prediction Model",
    ("ANN", "SVM", "KNN")
)

st.sidebar.markdown("---")
st.sidebar.info(
    "This app predicts whether a customer will churn "
    "based on their subscription and service details."
)

# ===============================
# Main Title
# ===============================
st.title("📊 Customer Churn Prediction System")
st.markdown(
    """
    This Streamlit web application uses **Machine Learning models**
    trained on the Telco Customer Churn dataset to predict
    whether a customer is likely to churn.
    """
)

# ===============================
# User Input Section
# ===============================
st.header("🧾 Customer Information")

col1, col2, col3 = st.columns(3)

with col1:
    gender = st.selectbox("Gender", ["Male", "Female"])
    SeniorCitizen = st.selectbox("Senior Citizen", [0, 1])
    Partner = st.selectbox("Partner", ["Yes", "No"])
    Dependents = st.selectbox("Dependents", ["Yes", "No"])
    tenure = st.slider("Tenure (Months)", 0, 72, 12)

with col2:
    PhoneService = st.selectbox("Phone Service", ["Yes", "No"])
    PaperlessBilling = st.selectbox("Paperless Billing", ["Yes", "No"])
    MonthlyCharges = st.number_input("Monthly Charges", 0.0, 200.0, 70.0)
    TotalCharges = st.number_input("Total Charges", 0.0, 10000.0, 1000.0)

with col3:
    Contract = st.selectbox(
        "Contract Type",
        ["Month-to-month", "One year", "Two year"]
    )
    InternetService = st.selectbox(
        "Internet Service",
        ["DSL", "Fiber optic", "No"]
    )
    PaymentMethod = st.selectbox(
        "Payment Method",
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)"
        ]
    )

# ===============================
# Create Input DataFrame
# ===============================
def prepare_input():
    data = {
        "gender": 1 if gender == "Male" else 0,
        "SeniorCitizen": SeniorCitizen,
        "Partner": 1 if Partner == "Yes" else 0,
        "Dependents": 1 if Dependents == "Yes" else 0,
        "tenure": tenure,
        "PhoneService": 1 if PhoneService == "Yes" else 0,
        "PaperlessBilling": 1 if PaperlessBilling == "Yes" else 0,
        "MonthlyCharges": MonthlyCharges,
        "TotalCharges": TotalCharges,
    }

    df_input = pd.DataFrame([data])

    # One-hot encoded columns (initialize all to 0)
    for col in feature_names:
        if col not in df_input.columns:
            df_input[col] = 0

    # Contract
    if Contract != "Month-to-month":
        df_input[f"Contract_{Contract}"] = 1

    # Internet Service
    if InternetService != "DSL":
        df_input[f"InternetService_{InternetService}"] = 1

    # Payment Method
    df_input[f"PaymentMethod_{PaymentMethod}"] = 1

    # Ensure correct column order
    df_input = df_input[feature_names]

    return df_input


# ===============================
# Prediction
# ===============================
st.markdown("---")
st.header("🔮 Prediction Result")

if st.button("Predict Churn"):

    input_df = prepare_input()

    # Scale input
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
            f"⚠️ **Customer is likely to churn**\n\n"
            f"Probability of churn: **{probability:.2%}**"
        )
    else:
        st.success(
            f"✅ **Customer is NOT likely to churn**\n\n"
            f"Probability of churn: **{probability:.2%}**"
        )

# ===============================
# Footer
# ===============================
st.markdown("---")
st.caption(
    "Customer Churn Prediction | ANN, SVM, KNN | Streamlit Deployment"
)

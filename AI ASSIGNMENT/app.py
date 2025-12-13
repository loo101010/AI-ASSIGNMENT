# ============================================================
# Customer Churn Prediction - Streamlit App (Corrected)
# ============================================================
# This app loads trained ML models (ANN, SVM, KNN) and predicts
# customer churn based on uploaded Telco customer data.
#
# Run with:
# streamlit run app.py
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import joblib
import os

# ============================================================
# App Configuration
# ============================================================
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Customer Churn Prediction App")
st.markdown(
    """
    This application predicts whether a customer will **churn**  
    using trained **ANN, SVM, and KNN models**.
    """
)

# ============================================================
# Load Saved Models and Objects with Error Handling
# ============================================================
@st.cache_resource
def load_models():
    # Check if files exist
    required_files = [
        "ann_model.joblib",
        "svm_model.joblib",
        "knn_model.joblib",
        "scaler.joblib",
        "feature_names.pkl"
    ]
    for file in required_files:
        if not os.path.exists(file):
            st.error(f"❌ Required file '{file}' not found!")
            st.stop()

    # Load models and preprocessing objects
    ann_model = joblib.load("ann_model.joblib")
    svm_model = joblib.load("svm_model.joblib")
    knn_model = joblib.load("knn_model.joblib")
    scaler = joblib.load("scaler.joblib")

    with open("feature_names.pkl", "rb") as f:
        feature_names = pickle.load(f)

    return ann_model, svm_model, knn_model, scaler, feature_names


ann_model, svm_model, knn_model, scaler, feature_names = load_models()
st.success("✅ Models and preprocessing objects loaded successfully!")

# ============================================================
# File Upload
# ============================================================
st.header("📁 Upload Customer Data")

uploaded_file = st.file_uploader(
    "Upload Telco Customer CSV file",
    type=["csv"]
)

# ============================================================
# Data Preprocessing Function
# ============================================================
def preprocess_data(df):
    df_processed = df.copy()

    # Drop customerID
    if "customerID" in df_processed.columns:
        df_processed.drop("customerID", axis=1, inplace=True)

    # Convert TotalCharges to numeric
    if "TotalCharges" in df_processed.columns:
        df_processed["TotalCharges"] = pd.to_numeric(
            df_processed["TotalCharges"], errors="coerce"
        )
        df_processed["TotalCharges"].fillna(0, inplace=True)

    # Encode binary variables
    binary_cols = [
        "gender", "Partner", "Dependents",
        "PhoneService", "PaperlessBilling"
    ]
    for col in binary_cols:
        if col in df_processed.columns:
            df_processed[col] = df_processed[col].map(
                {"Yes": 1, "No": 0, "Male": 1, "Female": 0}
            )

    # One-hot encode categorical variables
    categorical_cols = [
        "MultipleLines", "InternetService", "OnlineSecurity",
        "OnlineBackup", "DeviceProtection", "TechSupport",
        "StreamingTV", "StreamingMovies", "Contract",
        "PaymentMethod"
    ]
    df_processed = pd.get_dummies(
        df_processed, columns=categorical_cols, drop_first=True
    )

    # Ensure same feature order as training
    df_processed = df_processed.reindex(
        columns=feature_names, fill_value=0
    )

    return df_processed

# ============================================================
# Main Prediction Logic
# ============================================================
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.subheader("🔍 Uploaded Data Preview")
    st.dataframe(df.head())

    # Preprocess data
    X_processed = preprocess_data(df)

    # Scale features
    X_scaled = scaler.transform(X_processed)

    # Model selection
    st.subheader("🤖 Select Model")
    model_choice = st.selectbox(
        "Choose a trained model:",
        ["ANN", "SVM", "KNN"]
    )

    if model_choice == "ANN":
        model = ann_model
    elif model_choice == "SVM":
        model = svm_model
    else:
        model = knn_model

    # Predict
    predictions = model.predict(X_scaled)
    probabilities = model.predict_proba(X_scaled)[:, 1]

    # ========================================================
    # Display Results
    # ========================================================
    st.subheader("📊 Prediction Results")

    results_df = pd.DataFrame({
        "Prediction": np.where(predictions == 1, "Churn", "Not Churn"),
        "Churn Probability": probabilities.round(4)
    })

    st.dataframe(results_df)

    # Summary statistics
    churn_rate = (predictions.sum() / len(predictions)) * 100

    st.metric(
        label="📉 Predicted Churn Rate",
        value=f"{churn_rate:.2f}%"
    )

    st.success("✅ Prediction completed successfully!")

# ============================================================
# Footer
# ============================================================
st.markdown("---")
st.markdown(
    "📌 **Note:** This app uses the same preprocessing and models "
    "trained in the Jupyter Notebook. Ensure all required model files "
    "are in the same folder as this script."
)

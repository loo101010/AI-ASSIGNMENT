import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Page configuration
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #FF6B6B;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #4ECDC4;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #FF6B6B;
    }
    .stButton>button {
        width: 100%;
        background-color: #FF6B6B;
        color: white;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #E85A5A;
    }
    </style>
""", unsafe_allow_html=True)

# Load models and preprocessors
@st.cache_resource
def load_models():
    try:
        with open('ann_model.keras', 'rb') as f:
            ann_model = pickle.load(f)
        with open('svm_model.joblib', 'rb') as f:
            svm_model = pickle.load(f)
        with open('knn_model.joblib', 'rb') as f:
            knn_model = pickle.load(f)
        with open('scaler.joblib', 'rb') as f:
            scaler = pickle.load(f)
        with open('feature_names.pkl', 'rb') as f:
            feature_names = pickle.load(f)
        return ann_model, svm_model, knn_model, scaler, feature_names
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None, None, None, None

ann_model, svm_model, knn_model, scaler, feature_names = load_models()

# Title
st.markdown('<h1 class="main-header">📊 Customer Churn Prediction System</h1>', unsafe_allow_html=True)
st.markdown("---")

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/clouds/200/000000/customer.png", width=150)
    st.markdown("## 🎯 Navigation")
    page = st.radio("Select Page:", ["🏠 Home", "🔮 Prediction", "📈 Model Comparison", "ℹ️ About"])
    
    st.markdown("---")
    st.markdown("### 📊 Quick Stats")
    st.info("**Models Available:** 3")
    st.success("**Best Model:** ANN")
    st.warning("**Accuracy:** ~80%")

# ============================================================================
# HOME PAGE
# ============================================================================
if page == "🏠 Home":
    st.markdown('<h2 class="sub-header">Welcome to Customer Churn Prediction System</h2>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3>🤖 AI-Powered</h3>
            <p>Three powerful machine learning models working together</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h3>⚡ Real-Time</h3>
            <p>Get instant predictions for customer churn risk</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h3>📊 Accurate</h3>
            <p>High accuracy predictions with detailed insights</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown("### 🎯 Project Overview")
    st.write("""
    This system predicts whether a customer is likely to churn (leave the company) based on their profile and behavior.
    It uses three different machine learning algorithms:
    
    - **ANN (Artificial Neural Network)**: Deep learning model with multiple layers
    - **SVM (Support Vector Machine)**: Finds the optimal decision boundary
    - **KNN (K-Nearest Neighbors)**: Classifies based on similar customers
    """)
    
    st.markdown("### 📝 How to Use")
    st.write("""
    1. Navigate to **🔮 Prediction** page
    2. Enter customer information in the form
    3. Click **Predict Churn** button
    4. View predictions from all three models
    5. Check **📈 Model Comparison** to see which model performs best
    """)
    
    st.markdown("### 🎓 Team Members")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.info("**Member 1**\nANN Model Developer")
    with col2:
        st.info("**Member 2**\nSVM Model Developer")
    with col3:
        st.info("**Member 3**\nKNN Model Developer")

# ============================================================================
# PREDICTION PAGE
# ============================================================================
elif page == "🔮 Prediction":
    st.markdown('<h2 class="sub-header">Make a Prediction</h2>', unsafe_allow_html=True)
    
    if ann_model is None:
        st.error("⚠️ Models not loaded. Please ensure all model files are in the correct directory.")
    else:
        st.write("Fill in the customer information below to predict churn probability.")
        
        with st.form("prediction_form"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("#### 👤 Personal Information")
                gender = st.selectbox("Gender", ["Male", "Female"])
                senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"])
                partner = st.selectbox("Partner", ["No", "Yes"])
                dependents = st.selectbox("Dependents", ["No", "Yes"])
                
            with col2:
                st.markdown("#### 📱 Service Information")
                phone_service = st.selectbox("Phone Service", ["No", "Yes"])
                multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
                internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
                online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
                online_backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
                
            with col3:
                st.markdown("#### 💼 Account Information")
                device_protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
                tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
                streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
                streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])
                
            col4, col5 = st.columns(2)
            
            with col4:
                st.markdown("#### 📄 Contract Details")
                contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
                paperless_billing = st.selectbox("Paperless Billing", ["No", "Yes"])
                payment_method = st.selectbox("Payment Method", 
                    ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
                
            with col5:
                st.markdown("#### 💰 Financial Information")
                tenure = st.slider("Tenure (months)", 0, 72, 12)
                monthly_charges = st.number_input("Monthly Charges ($)", 0.0, 200.0, 50.0, 0.1)
                total_charges = st.number_input("Total Charges ($)", 0.0, 10000.0, 500.0, 1.0)
            
            submitted = st.form_submit_button("🔮 Predict Churn")
            
            if submitted:
                # Prepare input data
                input_data = {
                    'gender': 1 if gender == "Male" else 0,
                    'SeniorCitizen': 1 if senior_citizen == "Yes" else 0,
                    'Partner': 1 if partner == "Yes" else 0,
                    'Dependents': 1 if dependents == "Yes" else 0,
                    'tenure': tenure,
                    'PhoneService': 1 if phone_service == "Yes" else 0,
                    'PaperlessBilling': 1 if paperless_billing == "Yes" else 0,
                    'MonthlyCharges': monthly_charges,
                    'TotalCharges': total_charges,
                }
                
                # Create a dataframe with all features (including one-hot encoded)
                input_df = pd.DataFrame([input_data])
                
                # Add one-hot encoded features (set all to 0 initially)
                for feature in feature_names:
                    if feature not in input_df.columns:
                        input_df[feature] = 0
                
                # Set the appropriate one-hot encoded features to 1
                if multiple_lines != "No phone service":
                    if multiple_lines == "Yes":
                        if 'MultipleLines_Yes' in feature_names:
                            input_df['MultipleLines_Yes'] = 1
                
                if internet_service != "No":
                    if 'InternetService_Fiber optic' in feature_names and internet_service == "Fiber optic":
                        input_df['InternetService_Fiber optic'] = 1
                    elif 'InternetService_No' in feature_names and internet_service == "No":
                        input_df['InternetService_No'] = 1
                
                # Set other service features
                service_features = {
                    'OnlineSecurity': online_security,
                    'OnlineBackup': online_backup,
                    'DeviceProtection': device_protection,
                    'TechSupport': tech_support,
                    'StreamingTV': streaming_tv,
                    'StreamingMovies': streaming_movies
                }
                
                for service, value in service_features.items():
                    if value == "Yes" and f'{service}_Yes' in feature_names:
                        input_df[f'{service}_Yes'] = 1
                    elif value == "No internet service" and f'{service}_No internet service' in feature_names:
                        input_df[f'{service}_No internet service'] = 1
                
                # Contract
                if contract == "One year" and 'Contract_One year' in feature_names:
                    input_df['Contract_One year'] = 1
                elif contract == "Two year" and 'Contract_Two year' in feature_names:
                    input_df['Contract_Two year'] = 1
                
                # Payment Method
                payment_cols = {
                    'Credit card (automatic)': 'PaymentMethod_Credit card (automatic)',
                    'Electronic check': 'PaymentMethod_Electronic check',
                    'Mailed check': 'PaymentMethod_Mailed check'
                }
                if payment_method in payment_cols and payment_cols[payment_method] in feature_names:
                    input_df[payment_cols[payment_method]] = 1
                
                # Ensure correct column order
                input_df = input_df[feature_names]
                
                # Scale the input
                input_scaled = scaler.transform(input_df)
                
                # Make predictions
                ann_pred_proba = ann_model.predict_proba(input_scaled)[0]
                svm_pred_proba = svm_model.predict_proba(input_scaled)[0]
                knn_pred_proba = knn_model.predict_proba(input_scaled)[0]
                
                ann_churn_prob = ann_pred_proba[1] * 100
                svm_churn_prob = svm_pred_proba[1] * 100
                knn_churn_prob = knn_pred_proba[1] * 100
                
                avg_churn_prob = (ann_churn_prob + svm_churn_prob + knn_churn_prob) / 3
                
                st.markdown("---")
                st.markdown("### 🎯 Prediction Results")
                
                # Display results
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("🧠 ANN Model", f"{ann_churn_prob:.1f}%", 
                             "High Risk" if ann_churn_prob > 50 else "Low Risk")
                
                with col2:
                    st.metric("⚙️ SVM Model", f"{svm_churn_prob:.1f}%",
                             "High Risk" if svm_churn_prob > 50 else "Low Risk")
                
                with col3:
                    st.metric("📊 KNN Model", f"{knn_churn_prob:.1f}%",
                             "High Risk" if knn_churn_prob > 50 else "Low Risk")
                
                with col4:
                    st.metric("📈 Average", f"{avg_churn_prob:.1f}%",
                             "High Risk" if avg_churn_prob > 50 else "Low Risk")
                
                # Visualization
                st.markdown("### 📊 Churn Probability Comparison")
                
                fig = go.Figure()
                
                models = ['ANN', 'SVM', 'KNN', 'Average']
                probabilities = [ann_churn_prob, svm_churn_prob, knn_churn_prob, avg_churn_prob]
                colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
                
                fig.add_trace(go.Bar(
                    x=models,
                    y=probabilities,
                    marker_color=colors,
                    text=[f'{p:.1f}%' for p in probabilities],
                    textposition='outside',
                    textfont=dict(size=14, color='black', family='Arial Black')
                ))
                
                fig.add_hline(y=50, line_dash="dash", line_color="red", 
                             annotation_text="Risk Threshold (50%)")
                
                fig.update_layout(
                    title="Churn Probability by Model",
                    xaxis_title="Model",
                    yaxis_title="Churn Probability (%)",
                    yaxis_range=[0, 100],
                    height=400,
                    showlegend=False
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Risk assessment
                if avg_churn_prob > 70:
                    st.error("⚠️ **HIGH RISK**: This customer has a high probability of churning. Immediate action recommended!")
                elif avg_churn_prob > 50:
                    st.warning("⚡ **MEDIUM RISK**: This customer shows signs of potential churn. Consider retention strategies.")
                else:
                    st.success("✅ **LOW RISK**: This customer is likely to stay. Continue providing excellent service!")
                
                # Recommendations
                st.markdown("### 💡 Recommendations")
                if avg_churn_prob > 50:
                    st.write("""
                    - **Offer Loyalty Rewards**: Provide special discounts or perks
                    - **Improve Customer Service**: Reach out proactively to address concerns
                    - **Contract Upgrade**: Consider offering a longer-term contract with benefits
                    - **Personalized Communication**: Send targeted retention campaigns
                    """)
                else:
                    st.write("""
                    - **Continue Engagement**: Maintain regular communication
                    - **Upsell Opportunities**: Introduce additional services
                    - **Reward Loyalty**: Acknowledge their continued patronage
                    - **Gather Feedback**: Use their positive experience as testimonials
                    """)

# ============================================================================
# MODEL COMPARISON PAGE
# ============================================================================
elif page == "📈 Model Comparison":
    st.markdown('<h2 class="sub-header">Model Performance Comparison</h2>', unsafe_allow_html=True)
    
    # Model metrics (these should come from your trained models)
    metrics_data = {
        'Model': ['ANN', 'SVM', 'KNN'],
        'Accuracy': [0.8045, 0.7964, 0.7657],
        'Precision': [0.6556, 0.6524, 0.5880],
        'Recall': [0.5504, 0.5252, 0.5295],
        'F1-Score': [0.5985, 0.5822, 0.5574],
        'AUC-ROC': [0.8465, 0.8408, 0.8231]
    }
    
    df_metrics = pd.DataFrame(metrics_data)
    
    # Display metrics table
    st.markdown("### 📊 Performance Metrics")
    st.dataframe(df_metrics.set_index('Model').style.highlight_max(axis=0, color='lightgreen'), 
                 use_container_width=True)
    
    # Visualization
    st.markdown("### 📈 Visual Comparison")
    
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC']
    
    fig = go.Figure()
    
    colors_map = {'ANN': '#FF6B6B', 'SVM': '#4ECDC4', 'KNN': '#45B7D1'}
    
    for model in df_metrics['Model']:
        values = df_metrics[df_metrics['Model'] == model][metrics].values[0]
        fig.add_trace(go.Bar(
            name=model,
            x=metrics,
            y=values,
            marker_color=colors_map[model],
            text=[f'{v:.4f}' for v in values],
            textposition='outside'
        ))
    
    fig.update_layout(
        title="Model Performance Metrics Comparison",
        xaxis_title="Metrics",
        yaxis_title="Score",
        yaxis_range=[0, 1],
        barmode='group',
        height=500,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Best model highlight
    st.markdown("### 🏆 Best Model")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.info("**Best Accuracy**\n\n🥇 ANN (80.45%)")
    
    with col2:
        st.info("**Best F1-Score**\n\n🥇 ANN (59.85%)")
    
    with col3:
        st.info("**Best AUC-ROC**\n\n🥇 ANN (84.65%)")
    
    # Radar chart
    st.markdown("### 🎯 Radar Chart Comparison")
    
    fig_radar = go.Figure()
    
    for model in df_metrics['Model']:
        values = df_metrics[df_metrics['Model'] == model][metrics].values[0].tolist()
        values.append(values[0])  # Close the radar chart
        
        fig_radar.add_trace(go.Scatterpolar(
            r=values,
            theta=metrics + [metrics[0]],
            fill='toself',
            name=model,
            line_color=colors_map[model]
        ))
    
    fig_radar.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1])
        ),
        showlegend=True,
        height=500
    )
    
    st.plotly_chart(fig_radar, use_container_width=True)
    
    # Model descriptions
    st.markdown("### 📝 Model Descriptions")
    
    tab1, tab2, tab3 = st.tabs(["🧠 ANN", "⚙️ SVM", "📊 KNN"])
    
    with tab1:
        st.write("""
        **Artificial Neural Network (ANN)**
        
        - **Architecture**: Multi-layer perceptron with 100 and 50 neurons
        - **Activation**: ReLU (Rectified Linear Unit)
        - **Optimizer**: Adam
        - **Best For**: Complex patterns and non-linear relationships
        - **Performance**: Highest overall accuracy and AUC-ROC
        """)
    
    with tab2:
        st.write("""
        **Support Vector Machine (SVM)**
        
        - **Kernel**: RBF (Radial Basis Function)
        - **C Parameter**: 1.0
        - **Gamma**: Scale
        - **Best For**: High-dimensional spaces and clear margin separation
        - **Performance**: Good balance between precision and recall
        """)
    
    with tab3:
        st.write("""
        **K-Nearest Neighbors (KNN)**
        
        - **K Value**: 5 neighbors
        - **Weights**: Distance-weighted
        - **Metric**: Euclidean distance
        - **Best For**: Simple, instance-based learning
        - **Performance**: Fast prediction, good for real-time applications
        """)

# ============================================================================
# ABOUT PAGE
# ============================================================================
elif page == "ℹ️ About":
    st.markdown('<h2 class="sub-header">About This Project</h2>', unsafe_allow_html=True)
    
    st.write("""
    ### 🎯 Project Objective
    
    This Customer Churn Prediction System is designed to help businesses identify customers 
    who are at risk of leaving (churning). By predicting churn early, companies can take 
    proactive measures to retain valuable customers.
    
    ### 📊 Dataset
    
    The system is trained on the **Telco Customer Churn Dataset**, which includes:
    - Customer demographics (gender, age, partner, dependents)
    - Account information (tenure, contract, payment method)
    - Services used (phone, internet, security, backup, etc.)
    - Charges (monthly and total)
    
    ### 🤖 Models Used
    
    We implemented and compared three different machine learning algorithms:
    
    1. **Artificial Neural Network (ANN)**: A deep learning model that can capture complex patterns
    2. **Support Vector Machine (SVM)**: A powerful classifier for high-dimensional data
    3. **K-Nearest Neighbors (KNN)**: A simple yet effective instance-based learning algorithm
    
    ### 🛠️ Technology Stack
    
    - **Frontend**: Streamlit
    - **Machine Learning**: scikit-learn
    - **Deep Learning**: scikit-learn MLPClassifier
    - **Visualization**: Plotly
    - **Data Processing**: Pandas, NumPy
    
    ### 👥 Team
    
    - **Member 1**: ANN Model Development & Training
    - **Member 2**: SVM Model Development & Training  
    - **Member 3**: KNN Model Development & Training
    
    ### 📫 Contact
    
    For questions or feedback, please contact the development team.
    
    ### 📄 License
    
    This project is created for educational purposes.
    """)
    
    st.markdown("---")
    st.markdown("Made with ❤️ using Streamlit")

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #888; padding: 1rem;'>
        <p>Customer Churn Prediction System v1.0 | © 2024</p>
    </div>
""", unsafe_allow_html=True)
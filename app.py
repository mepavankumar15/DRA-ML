import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import os
import plotly.graph_objects as go

st.set_page_config(page_title="Diabetic Risk Assessment", layout="wide")

@st.cache_resource
def load_models():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, "models", "best_model.pkl")
    scaler_path = os.path.join(base_dir, "models", "scaler.pkl")
    info_path = os.path.join(base_dir, "models", "model_info.json")
    
    if not (os.path.exists(model_path) and os.path.exists(scaler_path) and os.path.exists(info_path)):
        return None, None, None
        
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    with open(info_path, 'r') as f:
        model_info = json.load(f)
        
    return model, scaler, model_info

model, scaler, model_info = load_models()

if model is None or scaler is None or model_info is None:
    st.error("Please run the notebook first to generate model files")
    st.stop()

# Sidebar
st.sidebar.title("Diabetic Risk Assessment")
st.sidebar.markdown("### Model Information")
st.sidebar.info(f"**Model:** {model_info.get('model_name', 'Unknown')}\n\n"
                f"**Accuracy:** {model_info.get('accuracy', 0):.2f}\n\n"
                f"**ROC-AUC:** {model_info.get('roc_auc', 0):.2f}")

st.title("Diabetic Risk Assessment Tool")
st.markdown("Enter your health metrics below to assess your risk for diabetes.")

col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Gender", options=["Female", "Male"], index=0, help="Biological sex")
    pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=0, help="Number of times pregnant")
    glucose = st.slider("Glucose", min_value=0, max_value=250, value=120, help="Plasma glucose concentration a 2 hours in an oral glucose tolerance test")
    blood_pressure = st.slider("Blood Pressure", min_value=0, max_value=150, value=70, help="Diastolic blood pressure (mm Hg)")
    skin_thickness = st.slider("Skin Thickness", min_value=0, max_value=100, value=20, help="Triceps skin fold thickness (mm)")

with col2:
    insulin = st.slider("Insulin", min_value=0, max_value=900, value=79, help="2-Hour serum insulin (mu U/ml)")
    bmi = st.slider("BMI", min_value=0.0, max_value=70.0, value=25.0, help="Body mass index (weight in kg/(height in m)^2)")
    dpf = st.slider("Diabetes Pedigree Function", min_value=0.0, max_value=2.5, value=0.5, help="Diabetes pedigree function (genetic risk)")
    age = st.slider("Age", min_value=21, max_value=100, value=30, help="Age in years")

# Main content and prediction
if st.button("**Assess Risk**", type="primary"):
    gender_val = 1 if gender == "Female" else 0
    input_data = pd.DataFrame([[
        gender_val, pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age
    ]], columns=['Gender', 'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'])
    
    # Scale features
    scaled_data = scaler.transform(input_data)
    
    # Predict
    prediction = model.predict(scaled_data)[0]
    probability = model.predict_proba(scaled_data)[0][1] * 100
    
    st.markdown("---")
    st.subheader("Assessment Results")
    
    # Categorize risk
    if probability < 30:
        risk_level = "Low"
        st.success(f"Risk Level: **{risk_level}** ({probability:.1f}%)")
    elif probability <= 60:
        risk_level = "Medium"
        st.warning(f"Risk Level: **{risk_level}** ({probability:.1f}%)")
    else:
        risk_level = "High"
        st.error(f"Risk Level: **{risk_level}** ({probability:.1f}%)")
        
    col_chart1, col_chart2 = st.columns(2)
    
    with col_chart1:
        # Gauge chart
        fig_gauge = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = probability,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Diabetes Risk Probability"},
            gauge = {
                'axis': {'range': [None, 100]},
                'bar': {'color': "black"},
                'steps': [
                    {'range': [0, 30], 'color': "lightgreen"},
                    {'range': [30, 60], 'color': "yellow"},
                    {'range': [60, 100], 'color': "red"}
                ],
            }
        ))
        st.plotly_chart(fig_gauge, use_container_width=True)
        
    with col_chart2:
        # Feature importances chart
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importances = np.abs(model.coef_[0])
        else:
            importances = np.ones(9) / 9 # fallback
            
        features = ['Gender', 'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DPF', 'Age']
        
        # Sort features by importance
        indices = np.argsort(importances)
        
        fig_bar = go.Figure(go.Bar(
            x=importances[indices],
            y=[features[i] for i in indices],
            orientation='h'
        ))
        fig_bar.update_layout(title="Feature Importance in Model", margin=dict(l=0, r=0, t=30, b=0))
        st.plotly_chart(fig_bar, use_container_width=True)
        
    st.markdown("### What This Means")
    if risk_level == "Low":
        st.info("Based on the model, your metrics indicate a low probability of diabetes. Continue maintaining a healthy lifestyle, a balanced diet, and regular exercise.")
    elif risk_level == "Medium":
        st.warning("Your metrics show a moderate risk. You should consider discussing these results with a healthcare provider and monitor your diet and physical activity.")
    else:
        st.error("Your metrics suggest a high probability of diabetes. It is highly recommended to consult a doctor or healthcare professional for a complete medical checkup.")

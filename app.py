"""
app.py
Streamlit web interface for the Diabetic Risk Assessment application.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from predictor import predict, get_shap_chart, get_risk_gauge, get_feature_status_df, init_models

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="Diabetic Risk Assessment",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize models
init_models()

# --- CUSTOM CSS ---
st.markdown("""
<style>
    /* Clean white background and sans-serif font */
    body {
        font-family: sans-serif;
        background-color: #FFFFFF;
    }
    .risk-card {
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        padding: 20px;
        margin: 10px 0;
        background-color: #FFFFFF;
    }
    .metric-box {
        background-color: #F5F5F5;
        border-radius: 8px;
        text-align: center;
        padding: 10px;
    }
    .risk-badge {
        display: inline-block;
        padding: 10px 20px;
        border-radius: 50px;
        font-size: 1.2rem;
        font-weight: bold;
        color: white;
        margin: 10px 0;
    }
    .normal-tag {
        background-color: #E8F5E9;
        color: #2E7D32;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 0.85rem;
        font-weight: bold;
    }
    .warning-tag {
        background-color: #FFF3E0;
        color: #E65100;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 0.85rem;
        font-weight: bold;
    }
    .header-strip {
        width: 100%;
        height: 5px;
        background: linear-gradient(90deg, #2E7D32, #F57F17, #B71C1C);
        margin-bottom: 20px;
    }
    /* Remove Streamlit default padding */
    #MainMenu {visibility:hidden;}
    footer {visibility:hidden;}
    header {visibility:hidden;}
</style>
""", unsafe_allow_html=True)

# --- SIDEBAR ---
st.sidebar.markdown("## 🩺 Diabetic Risk Assessment")
st.sidebar.markdown(
    "Enter your health vitals below. The ML model will compute your diabetes "
    "risk score, explain the key drivers, and flag any out-of-range values."
)
st.sidebar.markdown("---")
st.sidebar.markdown("**How it works:**")
st.sidebar.markdown("- Gradient Boosting + XGBoost ensemble predicts risk")
st.sidebar.markdown("- SHAP explains which vitals drive your score")
st.sidebar.markdown("- Normal ranges flag values needing attention")
st.sidebar.divider()
st.sidebar.markdown("**Models used:**\n- Gradient Boosting Classifier (60% weight)\n- XGBoost Classifier (40% weight)\n- SHAP for explainability")

# --- MAIN LAYOUT ---
col1, col2 = st.columns([2, 3])

# --- LEFT COLUMN: INPUT FORM ---
with col1:
    st.markdown("### 📋 Your Health Vitals")
    
    gender = st.selectbox(
        label="Gender", 
        options=["Female", "Male"], 
        help="Select your gender. This determines whether pregnancy-related vitals are considered."
    )
    
    if gender == "Female":
        pregnancies = st.number_input(
            label="Number of Pregnancies", min_value=0, max_value=20, value=1, step=1,
            help="Number of times pregnant. Enter 0 if never pregnant."
        )
    else:
        # Males cannot be pregnant
        pregnancies = 0
        
    glucose = st.number_input(
        label="Glucose Level (mg/dL)", min_value=44, max_value=200, value=110, step=1,
        help="Plasma glucose concentration after a 2-hour oral glucose tolerance test. Normal fasting: 70-100 mg/dL."
    )
    
    blood_pressure = st.number_input(
        label="Blood Pressure (mm Hg)", min_value=24, max_value=122, value=72, step=1,
        help="Diastolic blood pressure. Normal range: 60-80 mm Hg."
    )
    
    skin_thickness = st.number_input(
        label="Skin Thickness (mm)", min_value=7, max_value=99, value=23, step=1,
        help="Triceps skinfold thickness — used to estimate body fat percentage."
    )
    
    insulin = st.number_input(
        label="Insulin Level (μU/mL)", min_value=14, max_value=846, value=80, step=1,
        help="2-hour serum insulin level. Normal range: 16-166 μU/mL."
    )
    
    bmi = st.number_input(
        label="BMI (kg/m²)", min_value=14.0, max_value=67.0, value=28.0, step=0.1,
        help="Body Mass Index. Normal: 18.5-24.9 | Overweight: 25-29.9 | Obese: 30+"
    )
    
    pedigree = st.number_input(
        label="Family History Score", min_value=0.0, max_value=2.5, value=0.35, step=0.01,
        help="Scores likelihood of diabetes based on family history. Higher = more family history of diabetes."
    )
    
    age = st.number_input(
        label="Age (years)", min_value=1, max_value=120, value=35, step=1,
        help="Your current age in years."
    )
    
    assess_btn = st.button("🔍 Assess My Risk", use_container_width=True, type="primary")

# --- RIGHT COLUMN: RESULTS ---
with col2:
    if not assess_btn:
        st.info("👈 Fill in your health vitals on the left and click 'Assess My Risk'")
    else:
        # Collect inputs
        inputs = {
            "Pregnancies": pregnancies,
            "Glucose": glucose,
            "BloodPressure": blood_pressure,
            "SkinThickness": skin_thickness,
            "Insulin": insulin,
            "BMI": bmi,
            "DiabetesPedigreeFunction": pedigree,
            "Age": age
        }
        
        try:
            # Run inference
            result = predict(inputs)
            risk_pct = result["risk_pct"]
            risk_info = result["risk_level"]
            
            # --- RESULT BLOCK 1: Risk Score + Gauge ---
            st.markdown("<div class='header-strip'></div>", unsafe_allow_html=True)
            res_col1, res_col2 = st.columns(2)
            
            with res_col1:
                fig_gauge = get_risk_gauge(risk_pct)
                st.pyplot(fig_gauge)
                
            with res_col2:
                st.metric("Diabetes Risk Score", f"{risk_pct}%")
                st.markdown(
                    f"<div class='risk-badge' style='background-color: {risk_info['color']};'>"
                    f"{risk_info['emoji']} {risk_info['label']}</div>", 
                    unsafe_allow_html=True
                )
                st.markdown(f"*{risk_info['message']}*")
                
                # Model agreement
                st.caption("Individual model scores:")
                m_col1, m_col2 = st.columns(2)
                m_col1.metric("Gradient Boosting", f"{result['gb_prob']}%")
                m_col2.metric("XGBoost", f"{result['xgb_prob']}%")
                
            st.divider()
            
            # --- RESULT BLOCK 2: SHAP Chart ---
            st.subheader("🧠 What's Driving Your Risk?")
            st.caption("Red bars increase risk. Blue bars decrease risk. Length = strength of impact.")
            fig_shap = get_shap_chart(result)
            st.pyplot(fig_shap)
            
            st.divider()
            
            # --- RESULT BLOCK 3: Feature Status Table ---
            st.subheader("📋 Your Vitals vs Normal Ranges")
            df_status = get_feature_status_df(result)
            
            # Define styling for Status column using pandas Styler
            def color_status(val):
                color = '#2E7D32' if 'Normal' in str(val) else '#E65100'
                background = '#E8F5E9' if 'Normal' in str(val) else '#FFF3E0'
                return f'color: {color}; background-color: {background}; font-weight: bold;'
                
            styled_df = df_status.style.map(color_status, subset=['Status'])
            st.dataframe(styled_df, use_container_width=True, hide_index=True)
            
            # --- RESULT BLOCK 4: Abnormal Flags ---
            if result["abnormal_flags"]:
                st.warning("⚠️ The following vitals are outside normal range:")
                
                tips_map = {
                    "Glucose": "Consider reducing sugar intake and getting a fasting glucose test.",
                    "BloodPressure": "Monitor blood pressure regularly. Reduce salt and stress.",
                    "BMI": "Regular physical activity and a calorie-conscious diet can help.",
                    "Insulin": "High insulin may indicate insulin resistance. Consult a doctor.",
                    "SkinThickness": "High skinfold may indicate elevated body fat percentage."
                }
                
                for flag in result["abnormal_flags"]:
                    feat = flag["feature"]
                    tip = tips_map.get(feat, "Please consult your healthcare provider.")
                    st.markdown(
                        f"- **{feat}**: {flag['value']} {flag['unit']} *(Normal: {flag['normal_range']} {flag['unit']})*\n  - *{tip}*"
                    )
            else:
                st.success("✅ All measured vitals are within normal range.")
                
            st.divider()
            
            # --- RESULT BLOCK 5: Risk Category Context ---
            st.subheader("📊 Risk Category Reference")
            
            # Horizontal color bar for context
            fig_bar, ax_bar = plt.subplots(figsize=(10, 0.5))
            fig_bar.patch.set_alpha(0)
            ax_bar.axis('off')
            
            colors = ['#2E7D32', '#558B2F', '#F57F17', '#E65100', '#B71C1C']
            ranges = [20, 20, 20, 20, 20]
            starts = [0, 20, 40, 60, 80]
            
            for i, (start, width, color) in enumerate(zip(starts, ranges, colors)):
                ax_bar.barh(0, width, left=start, color=color, height=1)
                
            # Marker for user's score
            ax_bar.plot(risk_pct, 0, marker='v', color='black', markersize=15)
            st.pyplot(fig_bar)
            
            ref_data = [
                {"Risk Level": "Very Low", "Range": "0-20%", "What It Means": "Healthy vitals, low genetic risk"},
                {"Risk Level": "Low", "Range": "20-40%", "What It Means": "Some risk factors present, monitor annually"},
                {"Risk Level": "Moderate", "Range": "40-60%", "What It Means": "Lifestyle changes recommended"},
                {"Risk Level": "High", "Range": "60-80%", "What It Means": "Medical consultation strongly advised"},
                {"Risk Level": "Very High", "Range": "80%+", "What It Means": "Immediate medical attention recommended"}
            ]
            st.dataframe(pd.DataFrame(ref_data), use_container_width=True, hide_index=True)
            
            st.divider()
            
            # --- RESULT BLOCK 6: Disclaimer ---
            st.markdown(
                "*⚕️ This tool is for educational and informational purposes only. "
                "It is not a medical diagnosis. Always consult a licensed healthcare "
                "provider for medical advice and testing.*"
            )
            
        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")

# --- FOOTER ---
st.divider()
st.caption("Built with scikit-learn · XGBoost · SHAP · Streamlit | Dataset: Pima Indians Diabetes (UCI)")

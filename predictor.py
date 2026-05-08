"""
predictor.py
Machine Learning inference module for Diabetic Risk Assessment.
Handles model loading, feature engineering, prediction, and visual generation (SHAP, gauge).
NO Streamlit imports allowed here.
"""

import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge
import shap

# --- CONSTANTS ---

FEATURES = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin",
    "BMI", "DiabetesPedigreeFunction", "Age", "GlucoseBMI", 
    "AgeInsulinRisk", "MetabolicScore"
]

FEATURE_LABELS = {
    "Pregnancies": "Number of Pregnancies",
    "Glucose": "Glucose Level (mg/dL)",
    "BloodPressure": "Blood Pressure (mm Hg)",
    "SkinThickness": "Skin Thickness (mm)",
    "Insulin": "Insulin Level (μU/mL)",
    "BMI": "Body Mass Index (BMI)",
    "DiabetesPedigreeFunction": "Family History Score",
    "Age": "Age (years)",
    "GlucoseBMI": "Glucose × BMI Risk",
    "AgeInsulinRisk": "Age-Insulin Risk Index",
    "MetabolicScore": "Composite Metabolic Score"
}

RISK_LEVELS = {
    (0, 20): {
        "label": "Very Low Risk", "color": "#2E7D32", "emoji": "✅", 
        "message": "Your vitals suggest very low diabetes risk. Keep up your healthy lifestyle."
    },
    (20, 40): {
        "label": "Low Risk", "color": "#558B2F", "emoji": "🟢", 
        "message": "Low risk detected. Maintain a balanced diet and regular exercise."
    },
    (40, 60): {
        "label": "Moderate Risk", "color": "#F57F17", "emoji": "🟡", 
        "message": "Moderate risk. Consider consulting a doctor and monitoring glucose levels."
    },
    (60, 80): {
        "label": "High Risk", "color": "#E65100", "emoji": "🟠", 
        "message": "High risk detected. Medical consultation is strongly recommended."
    },
    (80, 101): {
        "label": "Very High Risk", "color": "#B71C1C", "emoji": "🔴", 
        "message": "Very high risk. Please consult a doctor as soon as possible."
    }
}

NORMAL_RANGES = {
    "Glucose": (70, 140, "mg/dL"),
    "BloodPressure": (60, 90, "mm Hg"),
    "BMI": (18.5, 24.9, "kg/m²"),
    "Insulin": (16, 166, "μU/mL"),
    "SkinThickness": (10, 50, "mm"),
    "Age": (0, 120, "years"),
    "Pregnancies": (0, 20, "count"),
    "DiabetesPedigreeFunction": (0.0, 2.5, "score")
}

# --- MODEL LOADING ---
gb_model = None
xgb_model = None
scaler = None

def init_models():
    """Loads models from disk, or triggers training if they don't exist."""
    global gb_model, xgb_model, scaler
    if gb_model is not None:
        return
        
    gb_path = "models/gb_model.pkl"
    xgb_path = "models/xgb_model.pkl"
    scaler_path = "models/scaler.pkl"
    
    if not (os.path.exists(gb_path) and os.path.exists(xgb_path) and os.path.exists(scaler_path)):
        import streamlit as st
        # Auto-run data preparation and training
        with st.spinner("Initial setup: Downloading data and training ML models (this only happens once)..."):
            import prepare_data
            import train_model
            prepare_data.main()
            train_model.train_models()
            
    gb_model = joblib.load(gb_path)
    xgb_model = joblib.load(xgb_path)
    scaler = joblib.load(scaler_path)

# --- FUNCTIONS ---

def compute_engineered_features(inputs: dict) -> dict:
    """
    Takes raw user inputs and adds engineered features.
    Returns the full dictionary with all 11 features.
    """
    res = inputs.copy()
    res["GlucoseBMI"] = (res["Glucose"] * res["BMI"]) / 100
    res["AgeInsulinRisk"] = res["Age"] * (1 / (res["Insulin"] + 1)) * 100
    res["MetabolicScore"] = (res["Glucose"] / 100) + (res["BMI"] / 10) + (res["Age"] / 50)
    return res

def predict(inputs: dict) -> dict:
    """
    Runs inference on user inputs. Computes engineered features, scales them,
    and runs ensemble prediction using GB and XGB models.
    """
    full_inputs = compute_engineered_features(inputs)
    
    # Build single-row DataFrame
    df_in = pd.DataFrame([full_inputs], columns=FEATURES)
    
    # Scale features
    features_scaled = scaler.transform(df_in)
    
    # Get probabilities
    gb_prob = gb_model.predict_proba(features_scaled)[:, 1][0]
    xgb_prob = xgb_model.predict_proba(features_scaled)[:, 1][0]
    
    # Ensemble prediction (weighted average)
    final_prob = 0.6 * gb_prob + 0.4 * xgb_prob
    risk_pct = round(final_prob * 100, 1)
    
    # Determine risk level
    risk_level = None
    for (low, high), level_info in RISK_LEVELS.items():
        if low <= risk_pct < high:
            risk_level = level_info
            break
    if risk_pct >= 100:
        risk_level = RISK_LEVELS[(80, 101)]
        
    prediction = 1 if risk_pct >= 50 else 0
    
    # Identify abnormal flags
    abnormal_flags = []
    for feature, (min_val, max_val, unit) in NORMAL_RANGES.items():
        val = inputs[feature]
        if not (min_val <= val <= max_val):
            abnormal_flags.append({
                "feature": feature,
                "value": val,
                "normal_range": f"{min_val} - {max_val}",
                "unit": unit
            })
            
    return {
        "risk_pct": risk_pct,
        "risk_level": risk_level,
        "prediction": prediction,
        "gb_prob": round(gb_prob * 100, 1),
        "xgb_prob": round(xgb_prob * 100, 1),
        "final_prob": final_prob,
        "features_df": df_in,
        "features_scaled": features_scaled,
        "abnormal_flags": abnormal_flags
    }

def get_shap_chart(result: dict) -> plt.Figure:
    """
    Generates a SHAP horizontal bar chart to explain model predictions.
    """
    explainer = shap.TreeExplainer(gb_model)
    shap_values = explainer.shap_values(result["features_scaled"])
    
    # For a binary GradientBoostingClassifier in sklearn, shap_values might be single array or list
    if isinstance(shap_values, list):
        vals = shap_values[1][0]
    else:
        vals = shap_values[0]
        
    features = result["features_df"].iloc[0]
    
    # Create DataFrame for plotting
    df_shap = pd.DataFrame({
        "Feature": FEATURES,
        "SHAP Value": vals,
        "Feature Value": features.values
    })
    
    # Map feature names to human labels
    df_shap["Feature Label"] = df_shap["Feature"].map(FEATURE_LABELS)
    
    # Sort by absolute impact
    df_shap["Abs SHAP"] = df_shap["SHAP Value"].abs()
    df_shap = df_shap.sort_values(by="Abs SHAP", ascending=True)
    
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_alpha(0)  # Transparent background
    ax.set_facecolor('none')
    
    colors = ['#B71C1C' if x > 0 else '#1565C0' for x in df_shap["SHAP Value"]]
    ax.barh(df_shap["Feature Label"], df_shap["SHAP Value"], color=colors)
    ax.set_title("What's driving your risk score?", fontsize=14, fontweight="bold")
    ax.set_xlabel("SHAP Value (Impact on Prediction)")
    
    # Add grid
    ax.grid(axis='x', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    return fig

def get_risk_gauge(risk_pct: float) -> plt.Figure:
    """
    Creates a semicircular gauge chart for the risk percentage.
    """
    fig, ax = plt.subplots(figsize=(6, 3))
    fig.patch.set_alpha(0)
    ax.set_facecolor('none')
    ax.axis('equal')
    ax.axis('off')
    
    # Background semi-circle
    bg_wedge = Wedge((0, 0), 1, 0, 180, width=0.3, color='#E0E0E0')
    ax.add_patch(bg_wedge)
    
    # Determine color based on risk level
    color = '#2E7D32' # default green
    for (low, high), level_info in RISK_LEVELS.items():
        if low <= risk_pct < high:
            color = level_info["color"]
            break
    if risk_pct >= 100:
        color = RISK_LEVELS[(80, 101)]["color"]
        
    # Fill semi-circle based on risk (0 risk is angle 180, 100 risk is angle 0)
    angle = 180 - (risk_pct / 100 * 180)
    # Ensure angle is bounded
    angle = max(0, min(180, angle))
    fill_wedge = Wedge((0, 0), 1, angle, 180, width=0.3, color=color)
    ax.add_patch(fill_wedge)
    
    # Draw needle
    theta = np.radians(angle)
    x = [0, 0.85 * np.cos(theta)]
    y = [0, 0.85 * np.sin(theta)]
    ax.plot(x, y, color='black', linewidth=3)
    ax.plot(0, 0, marker='o', color='black', markersize=10)
    
    # Text
    ax.text(0, -0.2, f"{risk_pct}%", ha='center', va='center', fontsize=24, fontweight='bold', color=color)
    
    plt.tight_layout()
    return fig

def get_feature_status_df(result: dict) -> pd.DataFrame:
    """
    Returns a DataFrame comparing user vitals with normal ranges.
    Includes only original 8 features.
    """
    data = []
    # Original 8 features from input
    orig_features = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"]
    
    for feature in orig_features:
        val = result["features_df"].iloc[0][feature]
        min_val, max_val, unit = NORMAL_RANGES[feature]
        
        status = "✅ Normal" if (min_val <= val <= max_val) else "⚠️ Check"
        
        # Round value for clean display
        if isinstance(val, float):
            val = round(val, 2)
            
        data.append({
            "Feature": FEATURE_LABELS[feature],
            "Your Value": f"{val} {unit}",
            "Normal Range": f"{min_val} - {max_val} {unit}",
            "Status": status
        })
        
    return pd.DataFrame(data)

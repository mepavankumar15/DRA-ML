import os
import nbformat as nbf

def create_full_code_notebook():
    nb = nbf.v4.new_notebook()
    
    cells = [
        nbf.v4.new_markdown_cell("# 🚀 End-to-End ML Pipeline (Full Code)\nThis notebook contains the complete, streamlined code to download the data, preprocess it, train the models, and make a prediction. No explanations, just the raw code."),
        nbf.v4.new_code_cell("""import pandas as pd
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

# 1. Download Data
print("Downloading Pima Indians Diabetes dataset...")
diabetes = fetch_openml(name="diabetes", version=1, as_frame=True, parser="auto")
df = diabetes.frame
df = df.rename(columns={"class": "Outcome"})
df["Outcome"] = df["Outcome"].map({"tested_positive": 1, "tested_negative": 0})
col_mapping = {
    'preg': 'Pregnancies', 'plas': 'Glucose', 'pres': 'BloodPressure', 
    'skin': 'SkinThickness', 'insu': 'Insulin', 'mass': 'BMI', 
    'pedi': 'DiabetesPedigreeFunction', 'age': 'Age'
}
df = df.rename(columns=col_mapping).apply(pd.to_numeric)

# 2. Handle Zeros & Impute Missing
cols_with_zeros = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
for col in cols_with_zeros:
    df[col] = df[col].replace(0, np.nan)
df[cols_with_zeros] = KNNImputer(n_neighbors=5).fit_transform(df[cols_with_zeros])

# 3. Clip Outliers
ranges = {'Glucose': (44, 200), 'BloodPressure': (24, 122), 'BMI': (14, 67), 'SkinThickness': (7, 99), 'Insulin': (14, 846)}
for col, (lower, upper) in ranges.items():
    df[col] = df[col].clip(lower=lower, upper=upper)

# 4. Feature Engineering
df['GlucoseBMI'] = (df['Glucose'] * df['BMI']) / 100
df['AgeInsulinRisk'] = df['Age'] * (1 / (df['Insulin'] + 1)) * 100
df['MetabolicScore'] = (df['Glucose'] / 100) + (df['BMI'] / 10) + (df['Age'] / 50)

# 5. Train/Test Split & SMOTE
X = df.drop('Outcome', axis=1)
y = df['Outcome']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X_scaled, y)
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42, stratify=y_res)

# 6. Train Models
gb_model = GradientBoostingClassifier(n_estimators=300, learning_rate=0.05, max_depth=4, random_state=42)
gb_model.fit(X_train, y_train)

xgb_model = XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.08, eval_metric='logloss', random_state=42)
xgb_model.fit(X_train, y_train)
print("Models trained successfully!")

# 7. Sample Prediction
sample = pd.DataFrame([{
    'Pregnancies': 1, 'Glucose': 110, 'BloodPressure': 72, 'SkinThickness': 23, 'Insulin': 80, 
    'BMI': 28.0, 'DiabetesPedigreeFunction': 0.35, 'Age': 35,
    'GlucoseBMI': (110 * 28) / 100, 'AgeInsulinRisk': 35 * (1 / (80 + 1)) * 100, 
    'MetabolicScore': (110 / 100) + (28.0 / 10) + (35 / 50)
}])
sample_scaled = scaler.transform(sample)

gb_prob = gb_model.predict_proba(sample_scaled)[:, 1][0]
xgb_prob = xgb_model.predict_proba(sample_scaled)[:, 1][0]
final_prob = 0.6 * gb_prob + 0.4 * xgb_prob

print(f"\\n--- Sample Patient Prediction ---")
print(f"Gradient Boosting Probability: {gb_prob:.2%}")
print(f"XGBoost Probability:           {xgb_prob:.2%}")
print(f"Final Ensemble Risk Score:     {final_prob:.2%}")""")
    ]
    
    nb['cells'] = cells
    
    os.makedirs('notebooks', exist_ok=True)
    with open('notebooks/03_ML_full_code.ipynb', 'w', encoding='utf-8') as f:
        nbf.write(nb, f)
    print("Created notebooks/03_ML_full_code.ipynb")

if __name__ == '__main__':
    create_full_code_notebook()

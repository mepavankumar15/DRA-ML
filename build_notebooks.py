import os
import nbformat as nbf

def create_eda_notebook():
    nb = nbf.v4.new_notebook()
    
    cells = [
        nbf.v4.new_markdown_cell("# 📊 Exploratory Data Analysis (EDA)\nThis notebook explores the Pima Indians Diabetes Dataset to understand distributions, correlations, and relationships before building ML models."),
        
        nbf.v4.new_markdown_cell("### 1. Load and inspect `models/diabetes_data.csv`\nFirst, we load the dataset and take a look at the data structure and types."),
        nbf.v4.new_code_cell("import pandas as pd\nimport numpy as np\nimport matplotlib.pyplot as plt\nimport seaborn as sns\n\n# Load data\ndf = pd.read_csv('../models/diabetes_data.csv')\n\n# Display first few rows and info\ndisplay(df.head())\nprint(df.info())"),
        
        nbf.v4.new_markdown_cell("### 2. Class distribution bar chart\nUnderstanding the target class distribution helps us know if the dataset is imbalanced. Imbalanced datasets might require SMOTE or different class weights."),
        nbf.v4.new_code_cell("plt.figure(figsize=(6, 4))\nsns.countplot(data=df, x='Outcome', palette='Set2')\nplt.title('Class Distribution (0: No Diabetes, 1: Diabetes)')\nplt.show()"),
        
        nbf.v4.new_markdown_cell("### 3. Feature histograms for all 8 original features\nWe look at the distribution of each feature to spot skewness or outliers."),
        nbf.v4.new_code_cell("original_features = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']\ndf[original_features].hist(figsize=(12, 10), bins=20, color='skyblue', edgecolor='black')\nplt.tight_layout()\nplt.show()"),
        
        nbf.v4.new_markdown_cell("### 4. Correlation heatmap\nHeatmaps help us identify highly correlated features which might cause multicollinearity, or see which features correlate most with the outcome."),
        nbf.v4.new_code_cell("plt.figure(figsize=(10, 8))\ncorr = df.corr()\nsns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)\nplt.title('Feature Correlation Heatmap')\nplt.show()"),
        
        nbf.v4.new_markdown_cell("### 5. Box plots: each feature vs Outcome\nBox plots illustrate how feature values differ between the Diabetic and Non-Diabetic groups, highlighting potential predictive power."),
        nbf.v4.new_code_cell("plt.figure(figsize=(15, 10))\nfor i, col in enumerate(original_features, 1):\n    plt.subplot(3, 3, i)\n    sns.boxplot(data=df, x='Outcome', y=col, palette='Set2')\n    plt.title(f'{col} by Outcome')\nplt.tight_layout()\nplt.show()"),
        
        nbf.v4.new_markdown_cell("### 6. Pairplot of top 4 features colored by Outcome\nVisualizing pairwise relationships among the top correlated features gives insight into how they interact linearly or non-linearly to separate the classes."),
        nbf.v4.new_code_cell("top_features = ['Glucose', 'BMI', 'Age', 'DiabetesPedigreeFunction', 'Outcome']\nsns.pairplot(df[top_features], hue='Outcome', palette='Set2', diag_kind='kde')\nplt.show()"),
        
        nbf.v4.new_markdown_cell("### 7. Summary of key findings\n- **Imbalance**: The dataset has more Non-Diabetic (Class 0) than Diabetic (Class 1) cases.\n- **Glucose**: The most critical indicator; higher glucose strongly correlates with diabetes.\n- **BMI & Age**: Also show strong separation between classes.\n- **Correlations**: Glucose, BMI, and Age have the highest linear correlation with the Outcome. Engineered features (like GlucoseBMI) should capture the interaction well.")
    ]
    
    nb['cells'] = cells
    
    os.makedirs('notebooks', exist_ok=True)
    with open('notebooks/01_EDA.ipynb', 'w', encoding='utf-8') as f:
        nbf.write(nb, f)
    print("Created notebooks/01_EDA.ipynb")


def create_ml_process_notebook():
    nb = nbf.v4.new_notebook()
    
    cells = [
        nbf.v4.new_markdown_cell("# 🧠 ML Process Pipeline\nThis notebook walks through the entire Machine Learning pipeline from raw data preprocessing to evaluating the final models and generating SHAP explanations."),
        
        nbf.v4.new_markdown_cell("### 1. Load and inspect data\nWe start by loading the dataset. We're using the dataset saved from `prepare_data.py`."),
        nbf.v4.new_code_cell("# Import libraries\nimport pandas as pd\nimport numpy as np\nfrom sklearn.model_selection import train_test_split\nfrom sklearn.preprocessing import StandardScaler\nfrom sklearn.ensemble import GradientBoostingClassifier\nfrom xgboost import XGBClassifier\nfrom sklearn.metrics import confusion_matrix, roc_curve, auc, f1_score, classification_report\nfrom imblearn.over_sampling import SMOTE\nimport matplotlib.pyplot as plt\nimport seaborn as sns\nimport shap\n\n# Load dataset\ndf = pd.read_csv('../models/diabetes_data.csv')\ndf.head()"),
        
        nbf.v4.new_markdown_cell("### 2. Preprocessing (zero handling, KNN imputation, clipping)\n*Note: Since we are using `models/diabetes_data.csv`, this was already done in `prepare_data.py`. If you want to see the raw preprocessing logic:*"),
        nbf.v4.new_code_cell("# Zeros were replaced with NaN\n# df['Glucose'] = df['Glucose'].replace(0, np.nan)\n# KNN Imputer was applied to fill NaNs based on 5 nearest neighbors\n# df[cols] = KNNImputer(n_neighbors=5).fit_transform(df[cols])\n# Values were clipped to biological maximums/minimums\n# df['BMI'] = df['BMI'].clip(14, 67)\nprint('Preprocessing previously applied to models/diabetes_data.csv')"),
        
        nbf.v4.new_markdown_cell("### 3. Feature engineering\nThese 3 new features provide our models with composite risk indicators."),
        nbf.v4.new_code_cell("# 1. GlucoseBMI: Combined metabolic risk score\n# df['GlucoseBMI'] = (df['Glucose'] * df['BMI']) / 100\n\n# 2. AgeInsulinRisk: Age-weighted insulin inefficiency\n# df['AgeInsulinRisk'] = df['Age'] * (1 / (df['Insulin'] + 1)) * 100\n\n# 3. MetabolicScore: Composite baseline risk\n# df['MetabolicScore'] = (df['Glucose'] / 100) + (df['BMI'] / 10) + (df['Age'] / 50)\nprint('Engineered features are already present in the DataFrame:')\ndf[['GlucoseBMI', 'AgeInsulinRisk', 'MetabolicScore']].head()"),
        
        nbf.v4.new_markdown_cell("### 4. Scaling with StandardScaler\nModels like Gradient Boosting are somewhat robust to scale, but scaling helps the models converge and levels the playing field for feature importance analysis."),
        nbf.v4.new_code_cell("# Separate features and target\nX = df.drop('Outcome', axis=1)\ny = df['Outcome']\n\n# Scale features\nscaler = StandardScaler()\nX_scaled = scaler.fit_transform(X)\nprint('Features scaled.')"),
        
        nbf.v4.new_markdown_cell("### 5. SMOTE for class balance\nBecause the dataset has roughly 65% non-diabetic and 35% diabetic cases, we use Synthetic Minority Over-sampling Technique (SMOTE) to create synthetic examples of the minority class. This prevents the model from being biased toward predicting 'No Diabetes'."),
        nbf.v4.new_code_cell("smote = SMOTE(random_state=42)\nX_res, y_res = smote.fit_resample(X_scaled, y)\n\nprint(f'Before SMOTE: {np.bincount(y)}')\nprint(f'After SMOTE: {np.bincount(y_res)}')\n\n# Train/Test Split\nX_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42, stratify=y_res)"),
        
        nbf.v4.new_markdown_cell("### 6. Train Gradient Boosting\nWe train a Gradient Boosting model with carefully chosen hyperparameters. This is our primary model (60% weight)."),
        nbf.v4.new_code_cell("gb_model = GradientBoostingClassifier(n_estimators=300, learning_rate=0.05, max_depth=4, random_state=42)\ngb_model.fit(X_train, y_train)\nprint('Gradient Boosting Trained.')"),
        
        nbf.v4.new_markdown_cell("### 7. Train XGBoost\nWe train an XGBoost model. This acts as our secondary model (40% weight) to smooth out variance via ensembling."),
        nbf.v4.new_code_cell("xgb_model = XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.08, eval_metric='logloss', random_state=42)\nxgb_model.fit(X_train, y_train)\nprint('XGBoost Trained.')"),
        
        nbf.v4.new_markdown_cell("### 8. Evaluate both with confusion matrix + ROC curve + F1\nWe use AUC-ROC (to measure distinction between classes) and F1 Score (harmonic mean of precision and recall) as our primary metrics."),
        nbf.v4.new_code_cell("gb_preds = gb_model.predict(X_test)\nxgb_preds = xgb_model.predict(X_test)\n\n# Confusion Matrix\ncm = confusion_matrix(y_test, gb_preds)\nsns.heatmap(cm, annot=True, fmt='d', cmap='Blues')\nplt.title('GB Confusion Matrix')\nplt.show()\n\n# ROC Curve\nfpr, tpr, _ = roc_curve(y_test, gb_model.predict_proba(X_test)[:, 1])\nroc_auc = auc(fpr, tpr)\nplt.plot(fpr, tpr, label=f'GB AUC = {roc_auc:.2f}')\nplt.plot([0, 1], [0, 1], 'k--')\nplt.legend(loc='lower right')\nplt.title('Receiver Operating Characteristic')\nplt.show()\n\nprint('GB F1 Score:', f1_score(y_test, gb_preds))\nprint('XGB F1 Score:', f1_score(y_test, xgb_preds))"),
        
        nbf.v4.new_markdown_cell("### 9. SHAP summary plot + waterfall plot for one patient\nSHAP (SHapley Additive exPlanations) breaks down exactly *why* a model made a specific prediction by calculating each feature's contribution."),
        nbf.v4.new_code_cell("explainer = shap.TreeExplainer(gb_model)\nshap_values = explainer.shap_values(X_test)\n\n# Summary Plot\nshap.summary_plot(shap_values, X_test, feature_names=X.columns)\n\n# For waterfall plot, we look at the first test patient\nexplanation = shap.Explanation(values=shap_values[0], base_values=explainer.expected_value[0], data=X_test[0], feature_names=X.columns)\nshap.waterfall_plot(explanation)"),
        
        nbf.v4.new_markdown_cell("### 10. End-to-end prediction demo with sample inputs\nHere we run a sample patient through the ensemble pipeline."),
        nbf.v4.new_code_cell("sample_patient = pd.DataFrame([{\n    'Pregnancies': 2,\n    'Glucose': 130,\n    'BloodPressure': 75,\n    'SkinThickness': 25,\n    'Insulin': 100,\n    'BMI': 28.5,\n    'DiabetesPedigreeFunction': 0.45,\n    'Age': 35,\n    'GlucoseBMI': (130 * 28.5) / 100,\n    'AgeInsulinRisk': 35 * (1 / (100 + 1)) * 100,\n    'MetabolicScore': (130 / 100) + (28.5 / 10) + (35 / 50)\n}])\n\n# Scale\nscaled_patient = scaler.transform(sample_patient)\n\n# Predict probabilities\ngb_prob = gb_model.predict_proba(scaled_patient)[:, 1][0]\nxgb_prob = xgb_model.predict_proba(scaled_patient)[:, 1][0]\nfinal_prob = 0.6 * gb_prob + 0.4 * xgb_prob\n\nprint(f'Gradient Boosting Probability: {gb_prob:.2%}')\nprint(f'XGBoost Probability: {xgb_prob:.2%}')\nprint(f'Ensemble Final Risk: {final_prob:.2%}')")
    ]
    
    nb['cells'] = cells
    
    with open('notebooks/02_ML_Process_explanation.ipynb', 'w', encoding='utf-8') as f:
        nbf.write(nb, f)
    print("Created notebooks/02_ML_Process_explanation.ipynb")

if __name__ == '__main__':
    create_eda_notebook()
    create_ml_process_notebook()

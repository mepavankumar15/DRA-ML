# 🩺 Diabetic Risk Assessment

A pure Machine Learning web application built with Streamlit, scikit-learn, and XGBoost that predicts a user's risk of diabetes based on their health vitals.

![App Screenshot](placeholder_screenshot.png) *(Add your screenshot here)*

## 🚀 Features
- **Ensemble ML Model**: Combines Gradient Boosting and XGBoost for robust predictions.
- **SHAP Explainability**: Visualizes exactly which vitals are driving the user's risk score.
- **Health Context**: Compares user inputs against medically normal ranges and flags abnormal values.
- **Auto-Training**: Automatically fetches data and trains models on first run.

## 🛠️ Tech Stack
- **Frontend**: Streamlit
- **Machine Learning**: scikit-learn, XGBoost, imbalanced-learn (SMOTE)
- **Explainability**: SHAP
- **Data processing & Visualization**: pandas, numpy, matplotlib, seaborn

## ⚙️ Local Setup

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd diabetes_checker
   ```

2. **Create and activate a virtual environment**
   ```bash
   # Windows
   python -m venv venv
   venv\Scripts\activate

   # Mac/Linux
   python -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the app**
   ```bash
   streamlit run app.py
   ```
   *(Note: The app will automatically download the dataset and train the models if they are not found).*

## 📊 Dataset
This project uses the **Pima Indians Diabetes Dataset** from the National Institute of Diabetes and Digestive and Kidney Diseases. The dataset is fetched automatically via the OpenML API.

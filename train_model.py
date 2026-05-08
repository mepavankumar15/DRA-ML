import os
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, f1_score, roc_auc_score, precision_score, recall_score
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

def train_models():
    """
    Trains the Gradient Boosting and XGBoost models, scales features, applies SMOTE,
    and saves the models and scaler.
    """
    # STEP 1: Load cleaned data
    print("Loading cleaned dataset...")
    df = pd.read_csv("models/diabetes_data.csv")
    
    # Define features and target
    FEATURES = [col for col in df.columns if col != "Outcome"]
    X = df[FEATURES]
    y = df["Outcome"]
    
    # STEP 2: Scale features
    print("Applying StandardScaler...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    os.makedirs("models", exist_ok=True)
    joblib.dump(scaler, "models/scaler.pkl")
    
    # STEP 3: Apply SMOTE
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X_scaled, y)
    
    print(f"Before SMOTE — Class 0: {sum(y == 0)}  |  Class 1: {sum(y == 1)}")
    print(f"After  SMOTE — Class 0: {sum(y_res == 0)}  |  Class 1: {sum(y_res == 1)}")
    
    # STEP 4: Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_res, y_res, test_size=0.2, random_state=42, stratify=y_res
    )
    print(f"Train: {len(X_train)} samples  |  Test: {len(X_test)} samples\n")
    
    # STEP 5: Train Model A: Gradient Boosting Classifier
    print("=== Gradient Boosting ===")
    gb_model = GradientBoostingClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=4,
        min_samples_split=10,
        min_samples_leaf=5,
        subsample=0.8,
        random_state=42
    )
    gb_model.fit(X_train, y_train)
    gb_preds = gb_model.predict(X_test)
    gb_probs = gb_model.predict_proba(X_test)[:, 1]
    
    print(classification_report(y_test, gb_preds, target_names=["Not Diabetic", "Diabetic"]))
    gb_f1 = f1_score(y_test, gb_preds)
    gb_auc = roc_auc_score(y_test, gb_probs)
    gb_prec = precision_score(y_test, gb_preds)
    gb_rec = recall_score(y_test, gb_preds)
    
    print(f"F1-Score:  {gb_f1:.3f}")
    print(f"AUC-ROC:   {gb_auc:.3f}\n")
    
    joblib.dump(gb_model, "models/gb_model.pkl")
    
    # STEP 6: Train Model B: XGBoost Classifier
    print("=== XGBoost ===")
    xgb_model = XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.08,
        subsample=0.8,
        colsample_bytree=0.8,
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42
    )
    xgb_model.fit(X_train, y_train)
    xgb_preds = xgb_model.predict(X_test)
    xgb_probs = xgb_model.predict_proba(X_test)[:, 1]
    
    print(classification_report(y_test, xgb_preds, target_names=["Not Diabetic", "Diabetic"]))
    xgb_f1 = f1_score(y_test, xgb_preds)
    xgb_auc = roc_auc_score(y_test, xgb_probs)
    xgb_prec = precision_score(y_test, xgb_preds)
    xgb_rec = recall_score(y_test, xgb_preds)
    
    joblib.dump(xgb_model, "models/xgb_model.pkl")
    
    # STEP 7: Cross-validation check
    scores = cross_val_score(gb_model, X_scaled, y, cv=5, scoring="f1")
    print("5-Fold CV F1 Scores:", np.round(scores, 3))
    print(f"Mean CV F1: {scores.mean():.3f}\n")
    
    # STEP 8: Print summary table
    print("=== Model Performance Summary ===")
    print(f"{'Model':<20} | {'F1-Score':<10} | {'Precision':<10} | {'Recall':<10} | {'AUC-ROC':<10}")
    print("-" * 70)
    print(f"{'Gradient Boosting':<20} | {gb_f1:<10.3f} | {gb_prec:<10.3f} | {gb_rec:<10.3f} | {gb_auc:<10.3f}")
    print(f"{'XGBoost':<20} | {xgb_f1:<10.3f} | {xgb_prec:<10.3f} | {xgb_rec:<10.3f} | {xgb_auc:<10.3f}")
    print("\nModels saved:")
    print("  gb_model.pkl   — " + str(os.path.getsize("models/gb_model.pkl") // 1024) + " KB" if os.path.exists("models/gb_model.pkl") else "  gb_model.pkl")
    print("  xgb_model.pkl  — " + str(os.path.getsize("models/xgb_model.pkl") // 1024) + " KB" if os.path.exists("models/xgb_model.pkl") else "  xgb_model.pkl")
    print("  scaler.pkl     — " + str(os.path.getsize("models/scaler.pkl") // 1024) + " KB" if os.path.exists("models/scaler.pkl") else "  scaler.pkl")
    print("All models saved ✅")

if __name__ == "__main__":
    train_models()

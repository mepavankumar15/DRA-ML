import os
import pandas as pd
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.impute import KNNImputer

def download_data() -> pd.DataFrame:
    """
    STEP 1: Download the dataset from OpenML.
    Returns a pandas DataFrame with standardized column names and numeric types.
    """
    print("Downloading Pima Indians Diabetes dataset...")
    # Fetch dataset from OpenML (id 37 or name 'diabetes' version 1)
    # Using parser='auto' to avoid warnings in newer sklearn versions
    diabetes = fetch_openml(name="diabetes", version=1, as_frame=True, parser="auto")
    df = diabetes.frame
    
    # Rename the target column and map to 0/1
    df = df.rename(columns={"class": "Outcome"})
    df["Outcome"] = df["Outcome"].map({"tested_positive": 1, "tested_negative": 0})
    
    # Standardize feature column names
    col_mapping = {
        'preg': 'Pregnancies',
        'plas': 'Glucose',
        'pres': 'BloodPressure',
        'skin': 'SkinThickness',
        'insu': 'Insulin',
        'mass': 'BMI',
        'pedi': 'DiabetesPedigreeFunction',
        'age': 'Age'
    }
    df = df.rename(columns=col_mapping)
    
    # Ensure all columns are numeric
    df = df.apply(pd.to_numeric)
    
    print(f"Dataset loaded: {df.shape[0]} rows × {df.shape[1]} columns")
    return df

def handle_zeros(df: pd.DataFrame) -> pd.DataFrame:
    """
    STEP 2: Handle biologically impossible zeros by replacing them with NaN.
    """
    cols_with_zeros = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    print("Replacing biological zeros with NaN:")
    
    for col in cols_with_zeros:
        # Count zeros before replacing
        zero_count = (df[col] == 0).sum()
        # Replace 0 with NaN
        df[col] = df[col].replace(0, np.nan)
        # Format printing to match the expected output
        print(f"  {col}: {zero_count:>15} zeros replaced")
        
    return df

def impute_missing(df: pd.DataFrame) -> pd.DataFrame:
    """
    STEP 3: Use KNNImputer to fill missing values for the biological columns.
    """
    print("Running KNN Imputation (n_neighbors=5)...")
    cols_to_impute = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    
    imputer = KNNImputer(n_neighbors=5)
    
    # Fit and transform only the columns that have missing values
    imputed_values = imputer.fit_transform(df[cols_to_impute])
    
    df_imputed = df.copy()
    df_imputed[cols_to_impute] = imputed_values
    
    return df_imputed

def clip_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """
    STEP 4: Clip outliers to medically sensible ranges.
    """
    print("Clipping outliers to medical ranges...")
    ranges = {
        'Glucose': (44, 200),
        'BloodPressure': (24, 122),
        'BMI': (14, 67),
        'SkinThickness': (7, 99),
        'Insulin': (14, 846)
    }
    
    for col, (lower, upper) in ranges.items():
        df[col] = df[col].clip(lower=lower, upper=upper)
        
    return df

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    STEP 5: Add 3 new engineered features.
    """
    print("Adding engineered features: GlucoseBMI, AgeInsulinRisk, MetabolicScore")
    
    df['GlucoseBMI'] = (df['Glucose'] * df['BMI']) / 100
    df['AgeInsulinRisk'] = df['Age'] * (1 / (df['Insulin'] + 1)) * 100
    df['MetabolicScore'] = (df['Glucose'] / 100) + (df['BMI'] / 10) + (df['Age'] / 50)
    
    # Print the first 3 rows showing new features as requested
    print("\nFirst 3 rows showing new features:")
    print(df[['GlucoseBMI', 'AgeInsulinRisk', 'MetabolicScore']].head(3))
    
    return df

def main():
    """
    Main function to execute the data preparation pipeline.
    """
    # Run the pipeline steps
    df = download_data()
    df = handle_zeros(df)
    df = impute_missing(df)
    df = clip_outliers(df)
    df = engineer_features(df)
    
    # STEP 6: Save cleaned data
    os.makedirs("models", exist_ok=True)
    save_path = "models/diabetes_data.csv"
    df.to_csv(save_path, index=False)
    
    # Print final summary
    print(f"\nFinal dataset shape: {df.shape}")
    print("Class distribution:")
    class_counts = df['Outcome'].value_counts()
    total = len(df)
    print(f"  Not Diabetic (0): {class_counts[0]} ({(class_counts[0]/total)*100:.1f}%)")
    print(f"  Diabetic     (1): {class_counts[1]} ({(class_counts[1]/total)*100:.1f}%)")
    print(f"Data ready ✅  →  {save_path}")

if __name__ == "__main__":
    main()

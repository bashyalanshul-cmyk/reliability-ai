import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from config.config import MODEL_DIR, TARGET_COLUMN, RAW_DATA_DIR
from src.data.generator import generate_baseline_data

def preprocess_data(df, preprocessor=None, fit=False):
    """Preprocess data using column transformer"""
    df = df.copy()
    df = df.drop(['CustomerID'], axis=1, errors='ignore')
    
    numerical_cols = ['Tenure', 'MonthlyCharges', 'TotalCharges', 'SeniorCitizen']
    categorical_cols = ['Gender', 'Partner', 'Dependents', 'InternetService', 
                       'Contract', 'PaperlessBilling', 'PaymentMethod']
    
    if preprocessor is None:
        numerical_transformer = StandardScaler()
        categorical_transformer = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, numerical_cols),
                ('cat', categorical_transformer, categorical_cols)
            ])
    
    if fit:
        X = preprocessor.fit_transform(df.drop(TARGET_COLUMN, axis=1, errors='ignore'))
    else:
        X = preprocessor.transform(df.drop(TARGET_COLUMN, axis=1, errors='ignore'))
    
    y = df[TARGET_COLUMN] if TARGET_COLUMN in df.columns else None
    return X, y, preprocessor

def train_model():
    """Train baseline classification model"""
    df = generate_baseline_data()
    
    X, y, preprocessor = preprocess_data(df, fit=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        random_state=42
    )
    model.fit(X_train, y_train)
    
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)
    
    metrics = {
        'roc_auc': roc_auc_score(y_test, y_pred_proba),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred)
    }
    
    # Save model and preprocessor
    joblib.dump(model, MODEL_DIR / 'churn_model.pkl')
    joblib.dump(preprocessor, MODEL_DIR / 'preprocessor.pkl')
    
    return model, preprocessor, metrics

def load_model():
    """Load trained model and preprocessor"""
    model = joblib.load(MODEL_DIR / 'churn_model.pkl')
    preprocessor = joblib.load(MODEL_DIR / 'preprocessor.pkl')
    return model, preprocessor

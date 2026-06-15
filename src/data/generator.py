import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from config.config import RAW_DATA_DIR

def generate_baseline_data(n_samples=5000):
    """Generate baseline customer churn dataset for training"""
    np.random.seed(42)
    
    data = {
        'CustomerID': [f'CUST{i:05d}' for i in range(1, n_samples+1)],
        'Tenure': np.random.randint(1, 72, n_samples),
        'MonthlyCharges': np.random.normal(65, 30, n_samples).clip(10, 150),
        'TotalCharges': None,
        'Gender': np.random.choice(['Male', 'Female'], n_samples),
        'SeniorCitizen': np.random.binomial(1, 0.15, n_samples),
        'Partner': np.random.choice(['Yes', 'No'], n_samples),
        'Dependents': np.random.choice(['Yes', 'No'], n_samples),
        'InternetService': np.random.choice(['Fiber optic', 'DSL', 'No'], n_samples, p=[0.45, 0.35, 0.2]),
        'Contract': np.random.choice(['Month-to-month', 'One year', 'Two year'], n_samples, p=[0.55, 0.25, 0.2]),
        'PaperlessBilling': np.random.choice(['Yes', 'No'], n_samples),
        'PaymentMethod': np.random.choice(
            ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'], 
            n_samples, p=[0.35, 0.25, 0.2, 0.2]
        ),
        'Churn': None
    }
    
    df = pd.DataFrame(data)
    df['TotalCharges'] = df['Tenure'] * df['MonthlyCharges'] * np.random.uniform(0.95, 1.05, n_samples)
    
    # Churn logic
    churn_prob = (
        0.05 +
        0.02 * (df['Tenure'] < 6) +
        0.15 * (df['Contract'] == 'Month-to-month') +
        0.1 * (df['InternetService'] == 'Fiber optic') +
        0.08 * (df['PaymentMethod'] == 'Electronic check') +
        0.05 * (df['SeniorCitizen'] == 1)
    )
    df['Churn'] = (np.random.rand(n_samples) < churn_prob).astype(int)
    
    df.to_csv(RAW_DATA_DIR / 'baseline_data.csv', index=False)
    return df

def generate_production_data(day=1, drift_level=0.0):
    """Generate daily production data with optional drift"""
    np.random.seed(42 + day)
    n_samples = np.random.randint(200, 400)
    
    data = {
        'CustomerID': [f'PROD{i:05d}' for i in range(1, n_samples+1)],
        'Tenure': None,
        'MonthlyCharges': None,
        'TotalCharges': None,
        'Gender': np.random.choice(['Male', 'Female'], n_samples),
        'SeniorCitizen': np.random.binomial(1, 0.15 + drift_level*0.5, n_samples),
        'Partner': np.random.choice(['Yes', 'No'], n_samples),
        'Dependents': np.random.choice(['Yes', 'No'], n_samples),
        'InternetService': None,
        'Contract': None,
        'PaperlessBilling': np.random.choice(['Yes', 'No'], n_samples),
        'PaymentMethod': None,
        'Churn': None
    }
    
    df = pd.DataFrame(data)
    
    # Introduce drift
    df['Tenure'] = np.random.randint(1, max(2, 72 - int(drift_level*60)), n_samples)
    df['MonthlyCharges'] = np.random.normal(65 + drift_level*80, 30, n_samples).clip(10, 250)
    
    isp_probs = np.array([0.45 + drift_level*0.5, 0.35 - drift_level*0.25, 0.2 - drift_level*0.25])
    isp_probs = np.clip(isp_probs, 0, None)
    isp_probs = isp_probs / isp_probs.sum()
    df['InternetService'] = np.random.choice(['Fiber optic', 'DSL', 'No'], n_samples, p=isp_probs)
    
    contract_probs = np.array([0.55 + drift_level*0.4, 0.25 - drift_level*0.2, 0.2 - drift_level*0.2])
    contract_probs = np.clip(contract_probs, 0, None)
    contract_probs = contract_probs / contract_probs.sum()
    df['Contract'] = np.random.choice(['Month-to-month', 'One year', 'Two year'], n_samples, p=contract_probs)
    
    pm_probs = np.array([0.35 + drift_level*0.5, 0.25, 0.2, 0.2 - drift_level*0.5])
    pm_probs = np.clip(pm_probs, 0, None)
    pm_probs = pm_probs / pm_probs.sum()
    df['PaymentMethod'] = np.random.choice(
        ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'], 
        n_samples, p=pm_probs
    )
    
    df['TotalCharges'] = df['Tenure'] * df['MonthlyCharges'] * np.random.uniform(0.95, 1.05, n_samples)
    
    # Churn logic with drift
    churn_prob = (
        0.05 + drift_level*0.3 +
        0.02 * (df['Tenure'] < 6) +
        0.15 * (df['Contract'] == 'Month-to-month') +
        0.1 * (df['InternetService'] == 'Fiber optic') +
        0.08 * (df['PaymentMethod'] == 'Electronic check') +
        0.05 * (df['SeniorCitizen'] == 1)
    )
    df['Churn'] = (np.random.rand(n_samples) < churn_prob).astype(int)
    
    return df

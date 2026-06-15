import pandas as pd
import numpy as np
from scipy.stats import ks_2samp
from config.config import DRIFT_THRESHOLD_KS, DRIFT_THRESHOLD_PSI

def calculate_psi(expected, actual, buckets=10):
    """Calculate Population Stability Index (PSI)"""
    expected = pd.Series(expected).dropna()
    actual = pd.Series(actual).dropna()
    
    if expected.empty or actual.empty:
        return 0.0
    
    try:
        breaks = np.linspace(expected.min(), expected.max(), buckets + 1)
        expected_counts = np.histogram(expected, breaks)[0]
        actual_counts = np.histogram(actual, breaks)[0]
    except:
        return 0.0
    
    expected_dist = expected_counts / len(expected)
    actual_dist = actual_counts / len(actual)
    
    expected_dist = np.where(expected_dist == 0, 0.0001, expected_dist)
    actual_dist = np.where(actual_dist == 0, 0.0001, actual_dist)
    
    psi_values = (expected_dist - actual_dist) * np.log(expected_dist / actual_dist)
    return psi_values.sum()

def detect_drift(baseline_df, current_df):
    """Detect data drift using KS test and PSI"""
    drift_report = {
        'timestamp': pd.Timestamp.now().isoformat(),
        'features': {},
        'overall_drift': False,
        'drifted_features': []
    }
    
    numerical_cols = ['Tenure', 'MonthlyCharges', 'TotalCharges', 'SeniorCitizen']
    
    for col in numerical_cols:
        if col in baseline_df.columns and col in current_df.columns:
            ks_stat, ks_pval = ks_2samp(baseline_df[col].dropna(), current_df[col].dropna())
            psi = calculate_psi(baseline_df[col], current_df[col])
            
            drifted = (ks_stat > DRIFT_THRESHOLD_KS) or (psi > DRIFT_THRESHOLD_PSI)
            
            drift_report['features'][col] = {
                'ks_statistic': float(ks_stat),
                'ks_p_value': float(ks_pval),
                'psi': float(psi),
                'drifted': bool(drifted)
            }
            
            if drifted:
                drift_report['drifted_features'].append(col)
    
    drift_report['overall_drift'] = len(drift_report['drifted_features']) > 0
    return drift_report

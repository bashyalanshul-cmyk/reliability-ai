import pandas as pd
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
from config.config import (
    PERFORMANCE_THRESHOLD_AUC,
    PERFORMANCE_THRESHOLD_PRECISION,
    PERFORMANCE_THRESHOLD_RECALL
)
from src.model.trainer import preprocess_data

def evaluate_model(model, preprocessor, df):
    """Evaluate model performance on new data"""
    X, y, _ = preprocess_data(df, preprocessor, fit=False)
    
    if y is None:
        return {'error': 'No target column available'}
    
    y_pred_proba = model.predict_proba(X)[:, 1]
    y_pred = model.predict(X)
    
    metrics = {
        'timestamp': pd.Timestamp.now().isoformat(),
        'roc_auc': float(roc_auc_score(y, y_pred_proba)),
        'precision': float(precision_score(y, y_pred)),
        'recall': float(recall_score(y, y_pred)),
        'f1': float(f1_score(y, y_pred)),
        'sample_size': len(df)
    }
    
    metrics['alerts'] = {
        'low_auc': metrics['roc_auc'] < PERFORMANCE_THRESHOLD_AUC,
        'low_precision': metrics['precision'] < PERFORMANCE_THRESHOLD_PRECISION,
        'low_recall': metrics['recall'] < PERFORMANCE_THRESHOLD_RECALL
    }
    
    metrics['performance_degraded'] = bool(any(metrics['alerts'].values()))
    return metrics

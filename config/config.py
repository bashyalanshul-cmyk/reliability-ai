import os
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODEL_DIR = DATA_DIR / "models"

os.makedirs(RAW_DATA_DIR, exist_ok=True)
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# Drift thresholds - lower to make it easier to detect
DRIFT_THRESHOLD_KS = 0.1  # KS statistic threshold
DRIFT_THRESHOLD_PSI = 0.15  # PSI threshold

# Performance thresholds
PERFORMANCE_THRESHOLD_AUC = 0.7
PERFORMANCE_THRESHOLD_PRECISION = 0.6
PERFORMANCE_THRESHOLD_RECALL = 0.5

# Target column
TARGET_COLUMN = "Churn"

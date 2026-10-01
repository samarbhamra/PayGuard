from __future__ import annotations

import os
from pathlib import Path

from pydantic import BaseModel


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
MLFLOW_DIR = PROJECT_ROOT / "mlruns"

XGB_MODEL_PATH = MODELS_DIR / "xgboost_prod_model.json"
LOGREG_MODEL_PATH = MODELS_DIR / "logreg_pipeline.joblib"
THRESHOLDS_PATH = MODELS_DIR / "thresholds.npz"
FEATURE_COLUMNS_PATH = MODELS_DIR / "feature_columns.txt"
DRIFT_REFERENCE_PATH = MODELS_DIR / "drift_reference.json"
METRICS_PATH = MODELS_DIR / "metrics.json"
ENSEMBLE_PATH = MODELS_DIR / "ensemble.json"

# Thresholds shared by labeling and feature engineering.
HIGH_AMOUNT = 500.0
VERY_HIGH_AMOUNT = 1000.0
HIGH_IP = 0.7
MEDIUM_IP = 0.4
HIGH_RISK_CATEGORIES = ("gaming", "travel")
START_TS = 1_735_689_600  # 2025-01-01 00:00:00 UTC
N_USERS = 20_000
PSI_ALERT = 0.25
MAX_REVIEW_RATE = 0.10


class TrainingConfig(BaseModel):
    n_samples: int = int(os.getenv("PAYGUARD_N_SAMPLES", "1000000"))
    fraud_rate: float = 0.02
    random_state: int = 42
    test_size: float = 0.3
    xgb_max_depth: int = 6
    xgb_learning_rate: float = 0.1
    xgb_n_estimators: int = 200


class ThresholdConfig(BaseModel):
    approve_threshold: float = 0.2
    flag_threshold: float = 0.6
    block_threshold: float = 0.9


TRAINING_CONFIG = TrainingConfig()
THRESHOLDS = ThresholdConfig()

DATA_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)
MLFLOW_DIR.mkdir(parents=True, exist_ok=True)

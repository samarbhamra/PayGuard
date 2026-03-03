from pathlib import Path
from pydantic import BaseModel


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
MLFLOW_DIR = PROJECT_ROOT / "mlruns"


class TrainingConfig(BaseModel):
    n_samples: int = 1_000_000
    fraud_rate: float = 0.02
    random_state: int = 42
    test_size: float = 0.2

    # XGBoost parameters (simplified)
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


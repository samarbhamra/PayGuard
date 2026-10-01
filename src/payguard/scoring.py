from __future__ import annotations

import numpy as np


def ensemble_proba(logreg, xgb_model, features: np.ndarray, weights: dict) -> np.ndarray:
    matrix = np.asarray(features, dtype=np.float32)
    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)
    p_logreg = logreg.predict_proba(matrix)[:, 1]
    p_xgb = xgb_model.predict_proba(matrix)[:, 1]
    return float(weights["logreg"]) * p_logreg + float(weights["xgboost"]) * p_xgb

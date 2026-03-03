from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import Depends, FastAPI
from sqlalchemy.orm import Session, sessionmaker
from xgboost import XGBClassifier

from ..config import MODELS_DIR
from .db import get_engine, init_db, log_scored_transaction
from .schemas import TransactionRequest, TransactionResponse

app = FastAPI(title="PayGuard Fraud Detection API", version="0.1.0")


def load_model() -> XGBClassifier:
    model_path = Path(os.getenv("PAYGUARD_MODEL_PATH", MODELS_DIR / "xgboost_prod_model.json"))
    model = XGBClassifier()
    model.load_model(model_path)
    return model


def load_thresholds():
    thr_path = Path(os.getenv("PAYGUARD_THRESHOLDS_PATH", MODELS_DIR / "thresholds.npz"))
    data = np.load(thr_path)
    return float(data["approve"]), float(data["flag"]), float(data["block"])


def load_feature_columns():
    cols_path = Path(os.getenv("PAYGUARD_FEATURE_COLUMNS_PATH", MODELS_DIR / "feature_columns.txt"))
    return [line.strip() for line in cols_path.read_text().splitlines() if line.strip()]


MODEL = load_model()
APPROVE_THR, FLAG_THR, BLOCK_THR = load_thresholds()
FEATURE_COLUMNS = load_feature_columns()

engine = init_db()
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@app.get("/health")
def health():
    return {"status": "ok"}


def build_feature_vector(req: TransactionRequest) -> np.ndarray:
    """
    Build a single-row feature vector that is compatible with the training pipeline.

    For simplicity, we reuse a minimal subset of feature transforms:
    - amount-based flags
    - hour_of_day derived features
    - one-hot encoding of high-cardinality categoricals using the saved feature columns.
    """
    base = {
        "user_id": req.user_id,
        "amount": req.amount,
        "ip_risk_score": req.ip_risk_score,
        "hour_of_day": req.hour_of_day,
        "is_night": int(req.hour_of_day < 6),
        "is_high_amount": int(req.amount > 500),
        "amount_bucket": (
            "low" if req.amount < 50 else "medium" if req.amount < 200 else "high" if req.amount < 500 else "very_high"
        ),
        "merchant_category": req.merchant_category,
        "device_type": req.device_type,
        "category_device_combo": f"{req.merchant_category}_{req.device_type}",
    }

    df = pd.DataFrame([base])

    # Manually one-hot encode matching the training dummies layout
    for col in ["merchant_category", "device_type", "amount_bucket", "category_device_combo"]:
        uniques = [c for c in FEATURE_COLUMNS if c.startswith(f"{col}_")]
        for u in uniques:
            val = u.split("_", 1)[1]
            df[u] = (df[col] == val).astype(int)
        df.drop(columns=[col], inplace=True)

    # Ensure all expected feature columns exist in the correct order
    for col in FEATURE_COLUMNS:
        if col not in df.columns:
            df[col] = 0

    df = df[FEATURE_COLUMNS]
    return df.values


def make_decision(prob: float) -> str:
    if prob >= BLOCK_THR:
        return "block"
    if prob >= FLAG_THR:
        return "flag"
    return "approve"


@app.post("/score", response_model=TransactionResponse)
def score_transaction(payload: TransactionRequest, db: Session = Depends(get_db)):
    X = build_feature_vector(payload)
    proba = float(MODEL.predict_proba(X)[0, 1])
    decision = make_decision(proba)

    log_scored_transaction(
        db,
        transaction_id=payload.transaction_id,
        user_id=payload.user_id,
        amount=payload.amount,
        merchant_category=payload.merchant_category,
        device_type=payload.device_type,
        ip_risk_score=payload.ip_risk_score,
        hour_of_day=payload.hour_of_day,
        fraud_probability=proba,
        decision=decision,
    )

    return TransactionResponse(
        transaction_id=payload.transaction_id,
        fraud_probability=proba,
        decision=decision,
        approve_threshold=APPROVE_THR,
        flag_threshold=FLAG_THR,
        block_threshold=BLOCK_THR,
    )


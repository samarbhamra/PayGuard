from __future__ import annotations

import json
import time

import joblib
import numpy as np
from fastapi import Depends, FastAPI, HTTPException
from sqlalchemy.orm import Session, sessionmaker
from xgboost import XGBClassifier

from ..config import (
    DRIFT_REFERENCE_PATH,
    ENSEMBLE_PATH,
    FEATURE_COLUMNS_PATH,
    LOGREG_MODEL_PATH,
    METRICS_PATH,
    PSI_ALERT,
    THRESHOLDS_PATH,
    XGB_MODEL_PATH,
)
from ..drift import drift_alert, feature_psi, load_reference, recalibrate_thresholds, score_psi
from ..features_online import build_model_vector
from ..scoring import ensemble_proba
from ..thresholds import make_decision
from . import db as database
from .db import get_user_history, init_db, log_scored_transaction, recent_transactions
from .schemas import TransactionRequest, TransactionResponse

app = FastAPI(title="PayGuard Fraud Detection API", version="0.2.0")

MIN_DRIFT_SAMPLES = 30


def _require(path, label: str):
    if not path.exists():
        raise RuntimeError(f"Missing {label} at {path}. Run: python -m src.payguard.train")
    return path


def load_runtime():
    _require(XGB_MODEL_PATH, "XGBoost model")
    _require(LOGREG_MODEL_PATH, "logistic regression model")
    _require(THRESHOLDS_PATH, "thresholds")
    _require(FEATURE_COLUMNS_PATH, "feature columns")
    _require(DRIFT_REFERENCE_PATH, "drift reference")
    xgb_model = XGBClassifier()
    xgb_model.load_model(XGB_MODEL_PATH)
    logreg = joblib.load(LOGREG_MODEL_PATH)
    weights = json.loads(ENSEMBLE_PATH.read_text()) if ENSEMBLE_PATH.exists() else {"logreg": 0.5, "xgboost": 0.5}
    saved = np.load(THRESHOLDS_PATH)
    columns = [line.strip() for line in FEATURE_COLUMNS_PATH.read_text().splitlines() if line.strip()]
    reference = load_reference(DRIFT_REFERENCE_PATH)
    metrics = json.loads(METRICS_PATH.read_text()) if METRICS_PATH.exists() else {}
    return xgb_model, logreg, weights, float(saved["approve"]), float(saved["flag"]), float(saved["block"]), columns, reference, metrics


XGB_MODEL, LOGREG_MODEL, WEIGHTS, APPROVE_THR, FLAG_THR, BLOCK_THR, FEATURE_COLUMNS, DRIFT_REF, TRAIN_METRICS = load_runtime()
engine = init_db()
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
print(
    f"PayGuard API ready. database={database.DATABASE_BACKEND} "
    f"ensemble_auc={TRAIN_METRICS.get('test', {}).get('roc_auc_ensemble')}"
)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def _live_drift(rows) -> tuple[bool, bool, float, float]:
    if len(rows) < MIN_DRIFT_SAMPLES:
        return False, False, FLAG_THR, BLOCK_THR
    scores = np.asarray([row.fraud_probability for row in rows], dtype=float)
    frame = {
        "amount": np.asarray([row.amount for row in rows], dtype=float),
        "ip_risk_score": np.asarray([row.ip_risk_score for row in rows], dtype=float),
        "hour_of_day": np.asarray([row.hour_of_day for row in rows], dtype=float),
    }
    feature_shift = feature_psi(frame, DRIFT_REF)
    score_shift = score_psi(scores, DRIFT_REF)
    alert = drift_alert(feature_shift, score_shift, alert_at=float(DRIFT_REF.get("psi_alert", PSI_ALERT)))
    flag_thr, block_thr = FLAG_THR, BLOCK_THR
    adjusted = False
    if score_shift >= float(DRIFT_REF.get("psi_alert", PSI_ALERT)):
        flag_thr, block_thr = recalibrate_thresholds(scores, DRIFT_REF["flag_rate"], DRIFT_REF["block_rate"])
        adjusted = True
    return alert, adjusted, flag_thr, block_thr


@app.get("/")
def root():
    return {
        "service": "PayGuard",
        "docs": "/docs",
        "health": "/health",
        "metrics": "/metrics",
        "drift": "/drift",
        "score": "/score",
    }


@app.get("/health")
def health():
    return {"status": "ok", "database": database.DATABASE_BACKEND, "model": "logreg+xgboost"}


@app.get("/metrics")
def metrics():
    if not TRAIN_METRICS:
        raise HTTPException(status_code=404, detail="Training metrics are not available")
    return TRAIN_METRICS


@app.get("/drift")
def drift_status(db: Session = Depends(get_db)):
    rows = recent_transactions(db, limit=500)
    live = None
    if len(rows) >= MIN_DRIFT_SAMPLES:
        scores = np.asarray([row.fraud_probability for row in rows], dtype=float)
        frame = {
            "amount": np.asarray([row.amount for row in rows], dtype=float),
            "ip_risk_score": np.asarray([row.ip_risk_score for row in rows], dtype=float),
            "hour_of_day": np.asarray([row.hour_of_day for row in rows], dtype=float),
        }
        feature_shift = feature_psi(frame, DRIFT_REF)
        score_shift = score_psi(scores, DRIFT_REF)
        live = {
            "samples": len(rows),
            "feature_psi": feature_shift,
            "score_psi": score_shift,
            "alert": drift_alert(feature_shift, score_shift, alert_at=float(DRIFT_REF.get("psi_alert", PSI_ALERT))),
        }
    return {
        "status": "warming_up" if live is None else ("alert" if live["alert"] else "ok"),
        "alert_threshold": float(DRIFT_REF.get("psi_alert", PSI_ALERT)),
        "samples": len(rows),
        "live": live,
        "training": TRAIN_METRICS.get("drift"),
    }


@app.post("/score", response_model=TransactionResponse)
def score_transaction(payload: TransactionRequest, db: Session = Depends(get_db)):
    now = time.time()
    history = get_user_history(db, payload.user_id, now)
    current = payload.model_dump()
    current["event_ts"] = now
    vector = build_model_vector(current, history, FEATURE_COLUMNS)
    proba = float(ensemble_proba(LOGREG_MODEL, XGB_MODEL, vector, WEIGHTS)[0])

    recent = recent_transactions(db)
    alert, adjusted, flag_thr, block_thr = _live_drift(recent)
    decision = make_decision(proba, flag_thr, block_thr)

    log_scored_transaction(
        db,
        transaction_id=payload.transaction_id,
        user_id=payload.user_id,
        amount=payload.amount,
        merchant_category=payload.merchant_category,
        device_type=payload.device_type,
        ip_risk_score=payload.ip_risk_score,
        hour_of_day=payload.hour_of_day,
        event_ts=now,
        fraud_probability=proba,
        decision=decision,
        drift_alert=alert,
    )

    return TransactionResponse(
        transaction_id=payload.transaction_id,
        fraud_probability=proba,
        decision=decision,
        approve_threshold=APPROVE_THR if not adjusted else flag_thr,
        flag_threshold=flag_thr,
        block_threshold=block_thr,
        drift_alert=alert,
        thresholds_adjusted_for_drift=adjusted,
    )

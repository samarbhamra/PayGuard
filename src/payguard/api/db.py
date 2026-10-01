from __future__ import annotations

import os
from datetime import datetime, timezone

from sqlalchemy import Column, DateTime, Float, Integer, String, create_engine, text
from sqlalchemy.orm import DeclarativeBase, Session

from ..config import DATA_DIR


class Base(DeclarativeBase):
    pass


class ScoredTransaction(Base):
    __tablename__ = "scored_transactions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    transaction_id = Column(String, nullable=False, index=True)
    user_id = Column(Integer, nullable=False, index=True)
    amount = Column(Float, nullable=False)
    merchant_category = Column(String, nullable=False)
    device_type = Column(String, nullable=False)
    ip_risk_score = Column(Float, nullable=False)
    hour_of_day = Column(Integer, nullable=False)
    event_ts = Column(Float, nullable=False)
    fraud_probability = Column(Float, nullable=False)
    decision = Column(String, nullable=False)
    drift_alert = Column(Integer, nullable=False, default=0)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)


DATABASE_BACKEND = "unconfigured"


def get_engine():
    global DATABASE_BACKEND
    env_url = os.getenv("PAYGUARD_DATABASE_URL")
    if env_url:
        DATABASE_BACKEND = "postgresql" if env_url.startswith("postgres") else env_url.split(":", 1)[0]
        connect_args = {"connect_timeout": 3} if env_url.startswith("postgres") else {}
        return create_engine(env_url, pool_pre_ping=True, connect_args=connect_args)

    pg_url = "postgresql://postgres:postgres@localhost:5432/payguard"
    try:
        engine = create_engine(pg_url, pool_pre_ping=True, connect_args={"connect_timeout": 2})
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        DATABASE_BACKEND = "postgresql"
        return engine
    except Exception:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        DATABASE_BACKEND = "sqlite"
        sqlite_path = DATA_DIR / "payguard.db"
        return create_engine(
            f"sqlite:///{sqlite_path}",
            connect_args={"check_same_thread": False},
        )


def init_db():
    engine = get_engine()
    Base.metadata.create_all(engine)
    return engine


def log_scored_transaction(
    session: Session,
    *,
    transaction_id: str,
    user_id: int,
    amount: float,
    merchant_category: str,
    device_type: str,
    ip_risk_score: float,
    hour_of_day: int,
    event_ts: float,
    fraud_probability: float,
    decision: str,
    drift_alert: bool,
):
    record = ScoredTransaction(
        transaction_id=transaction_id,
        user_id=user_id,
        amount=amount,
        merchant_category=merchant_category,
        device_type=device_type,
        ip_risk_score=ip_risk_score,
        hour_of_day=hour_of_day,
        event_ts=event_ts,
        fraud_probability=fraud_probability,
        decision=decision,
        drift_alert=int(drift_alert),
    )
    session.add(record)
    session.commit()
    return record


def get_user_history(session: Session, user_id: int, before_ts: float) -> list[dict]:
    rows = (
        session.query(ScoredTransaction)
        .filter(ScoredTransaction.user_id == user_id, ScoredTransaction.event_ts < before_ts)
        .all()
    )
    return [
        {
            "amount": row.amount,
            "merchant_category": row.merchant_category,
            "device_type": row.device_type,
            "ip_risk_score": row.ip_risk_score,
            "hour_of_day": row.hour_of_day,
            "event_ts": row.event_ts,
        }
        for row in rows
    ]


def recent_transactions(session: Session, limit: int = 400) -> list[ScoredTransaction]:
    return (
        session.query(ScoredTransaction).order_by(ScoredTransaction.id.desc()).limit(limit).all()
    )

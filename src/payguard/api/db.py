from __future__ import annotations

import os
from datetime import datetime

from sqlalchemy import Column, DateTime, Float, Integer, String, create_engine
from sqlalchemy.orm import DeclarativeBase, Session


class Base(DeclarativeBase):
    pass


class ScoredTransaction(Base):
    __tablename__ = "scored_transactions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    transaction_id = Column(String, nullable=False)
    user_id = Column(Integer, nullable=False)
    amount = Column(Float, nullable=False)
    merchant_category = Column(String, nullable=False)
    device_type = Column(String, nullable=False)
    ip_risk_score = Column(Float, nullable=False)
    hour_of_day = Column(Integer, nullable=False)
    fraud_probability = Column(Float, nullable=False)
    decision = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)


def get_engine():
    url = os.getenv("PAYGUARD_DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/payguard")
    return create_engine(url, pool_pre_ping=True)


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
    fraud_probability: float,
    decision: str,
):
    record = ScoredTransaction(
        transaction_id=transaction_id,
        user_id=user_id,
        amount=amount,
        merchant_category=merchant_category,
        device_type=device_type,
        ip_risk_score=ip_risk_score,
        hour_of_day=hour_of_day,
        fraud_probability=fraud_probability,
        decision=decision,
    )
    session.add(record)
    session.commit()




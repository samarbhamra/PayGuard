from __future__ import annotations

from pydantic import BaseModel, Field


class TransactionRequest(BaseModel):
    transaction_id: str = Field(..., description="Unique transaction identifier")
    user_id: int
    amount: float
    merchant_category: str
    device_type: str
    ip_risk_score: float
    hour_of_day: int


class TransactionResponse(BaseModel):
    transaction_id: str
    fraud_probability: float
    decision: str
    approve_threshold: float
    flag_threshold: float
    block_threshold: float




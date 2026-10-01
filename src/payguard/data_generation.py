from __future__ import annotations

import numpy as np
import pandas as pd

from .config import DATA_DIR, HIGH_AMOUNT, HIGH_RISK_CATEGORIES, N_USERS, START_TS, TRAINING_CONFIG


def _make_unique(timestamps: np.ndarray) -> np.ndarray:
    order = np.argsort(timestamps, kind="mergesort")
    sorted_ts = timestamps[order].astype(np.int64, copy=True)
    if len(sorted_ts) > 1 and np.any(np.diff(sorted_ts) <= 0):
        for i in range(1, len(sorted_ts)):
            if sorted_ts[i] <= sorted_ts[i - 1]:
                sorted_ts[i] = sorted_ts[i - 1] + 1
    out = np.empty_like(sorted_ts)
    out[order] = sorted_ts
    return out


def generate_synthetic_transactions(
    n_samples: int | None = None,
    fraud_rate: float | None = None,
    random_state: int | None = None,
    n_users: int | None = None,
) -> pd.DataFrame:
    """
    Synthetic card transactions with fraud patterns the model can learn:
    late night, risky merchant categories, high IP risk, large amounts,
    and short bursts of activity on a subset of users.
    """
    cfg = TRAINING_CONFIG
    n = n_samples or cfg.n_samples
    fraud_rate = fraud_rate if fraud_rate is not None else cfg.fraud_rate
    rng = np.random.default_rng(random_state or cfg.random_state)
    n_users = n_users or N_USERS
    span = 90 * 24 * 3600

    user_ids = rng.integers(1, n_users + 1, size=n)
    burst_user = rng.random(n_users + 1) < 0.08
    anchor = rng.integers(0, span - 7200, size=n_users + 1)
    is_burst = burst_user[user_ids]
    event_ts = np.where(
        is_burst,
        START_TS + anchor[user_ids] + rng.integers(0, 7200, size=n),
        START_TS + rng.integers(0, span, size=n),
    )
    event_ts = _make_unique(event_ts)

    merchant_categories = rng.choice(
        ["online_retail", "travel", "gaming", "food_delivery", "utilities"],
        size=n,
        p=[0.4, 0.15, 0.1, 0.25, 0.1],
    )
    device_types = rng.choice(["mobile", "desktop"], size=n, p=[0.7, 0.3])
    amounts = rng.lognormal(mean=4.2, sigma=1.05, size=n)
    hour_of_day = ((event_ts % 86400) // 3600).astype(int)
    ip_risk_score = rng.beta(a=2, b=5, size=n)

    night = (hour_of_day < 6).astype(float)
    risk_cat = np.isin(merchant_categories, list(HIGH_RISK_CATEGORIES)).astype(float)
    high_amt = (amounts > HIGH_AMOUNT).astype(float)
    burst = is_burst.astype(float)
    logit = (
        -7.0
        + 1.3 * night
        + 1.0 * risk_cat
        + 4.5 * ip_risk_score
        + 0.35 * (np.log1p(amounts) - 4.2)
        + 1.4 * high_amt
        + 1.2 * burst
        + 3.0 * night * ip_risk_score
        + 1.6 * risk_cat * high_amt
        + 2.2 * burst * (ip_risk_score > 0.5)
    )
    probability = 1.0 / (1.0 + np.exp(-np.clip(logit, -20, 20)))
    labels = rng.binomial(1, probability)

    return pd.DataFrame(
        {
            "transaction_id": [f"tx_{i}" for i in range(n)],
            "user_id": user_ids.astype(int),
            "amount": amounts.astype(float),
            "merchant_category": merchant_categories,
            "device_type": device_types,
            "ip_risk_score": ip_risk_score.astype(float),
            "hour_of_day": hour_of_day.astype(int),
            "event_ts": event_ts.astype(np.int64),
            "is_fraud": labels.astype(int),
        }
    )


def persist_raw_transactions(df: pd.DataFrame, path: str | None = None) -> str:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    out_path = DATA_DIR / (path or "transactions_raw.parquet")
    df.to_parquet(out_path, index=False)
    return str(out_path)


if __name__ == "__main__":
    df_ = generate_synthetic_transactions()
    path_ = persist_raw_transactions(df_)
    print(f"Wrote {len(df_):,} synthetic transactions to {path_}")

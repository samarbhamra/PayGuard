from __future__ import annotations

import numpy as np
import pandas as pd

from .config import DATA_DIR, TRAINING_CONFIG


def generate_synthetic_transactions(
    n_samples: int | None = None, fraud_rate: float | None = None, random_state: int | None = None
) -> pd.DataFrame:
    """
    Generate synthetic credit card transactions with simple but realistic patterns.

    The goal is not to perfectly simulate production data, but to create
    feature-rich data that exhibits clear fraud signals for modeling.
    """
    cfg = TRAINING_CONFIG
    n = n_samples or cfg.n_samples
    fraud_rate = fraud_rate if fraud_rate is not None else cfg.fraud_rate
    rng = np.random.default_rng(random_state or cfg.random_state)

    user_ids = rng.integers(1, 50_000, size=n)
    merchant_categories = rng.choice(
        ["online_retail", "travel", "gaming", "food_delivery", "utilities"], size=n, p=[0.4, 0.15, 0.1, 0.25, 0.1]
    )
    device_types = rng.choice(["mobile", "desktop"], size=n, p=[0.7, 0.3])

    amounts = rng.lognormal(mean=3.0, sigma=0.8, size=n)  # right-skewed
    hour_of_day = rng.integers(0, 24, size=n)
    ip_risk_score = rng.beta(a=2, b=8, size=n)  # mostly low risk

    # Base fraud probability with some engineered patterns:
    # - Higher at night (0-5)
    # - Higher for gaming / travel
    # - Higher for high ip_risk_score and high amounts
    base = np.full(n, fraud_rate)
    base += np.where(hour_of_day < 6, 0.03, 0.0)
    base += np.isin(merchant_categories, ["gaming", "travel"]) * 0.02
    base += (ip_risk_score > 0.7) * 0.05
    base += (amounts > 500) * 0.04
    base = np.clip(base, 0.001, 0.8)

    labels = rng.binomial(1, base)

    df = pd.DataFrame(
        {
            "transaction_id": [f"tx_{i}" for i in range(n)],
            "user_id": user_ids,
            "amount": amounts,
            "merchant_category": merchant_categories,
            "device_type": device_types,
            "ip_risk_score": ip_risk_score,
            "hour_of_day": hour_of_day,
            "is_fraud": labels,
        }
    )
    return df


def persist_raw_transactions(df: pd.DataFrame, path: str | None = None) -> str:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    out_path = DATA_DIR / (path or "transactions_raw.parquet")
    df.to_parquet(out_path, index=False)
    return str(out_path)


if __name__ == "__main__":
    df_ = generate_synthetic_transactions()
    path_ = persist_raw_transactions(df_)
    print(f"Wrote {len(df_):,} synthetic transactions to {path_}")


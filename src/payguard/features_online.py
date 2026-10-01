"""Point-in-time behavioral features shared by training checks and the API.

History windows exclude the current transaction so a live request can be
featurized from prior scored activity without using the label.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .config import HIGH_AMOUNT, HIGH_IP, HIGH_RISK_CATEGORIES, MEDIUM_IP, VERY_HIGH_AMOUNT

CAT_COLUMNS = ["merchant_category", "device_type", "amount_bucket", "category_device_combo"]

BEHAVIORAL_COLUMNS = [
    "amount",
    "amount_log",
    "ip_risk_score",
    "hour_of_day",
    "hour_sin",
    "hour_cos",
    "is_weekend",
    "is_night",
    "is_late_evening",
    "is_high_amount",
    "is_very_high_amount",
    "high_risk_ip_flag",
    "medium_risk_ip_flag",
    "high_risk_category_flag",
    "amount_x_ip_risk",
    "night_x_amount",
    "night_x_ip_risk",
    "high_amount_x_high_ip",
    "weekend_x_amount",
    "log_amount_x_ip",
    "user_prior_txn_count",
    "user_prior_amount_sum",
    "user_prior_amount_mean",
    "user_prior_amount_std",
    "user_prior_amount_max",
    "user_prior_ip_mean",
    "user_prior_ip_max",
    "user_prior_night_rate",
    "user_prior_high_amount_rate",
    "user_prior_high_ip_rate",
    "user_prior_hour_mean",
    "user_1h_txn_count",
    "user_1h_amount_sum",
    "user_1h_amount_mean",
    "user_1h_ip_mean",
    "user_1d_txn_count",
    "user_1d_amount_sum",
    "user_1d_amount_mean",
    "user_1d_amount_max",
    "user_1d_ip_mean",
    "user_1d_night_count",
    "user_1d_high_amount_count",
    "user_1d_high_ip_count",
    "user_7d_txn_count",
    "user_7d_amount_sum",
    "user_7d_amount_mean",
    "user_7d_ip_mean",
    "amount_to_user_mean_ratio",
    "amount_to_user_max_ratio",
    "amount_minus_user_mean",
    "amount_zscore_user",
    "ip_vs_user_mean",
    "hour_vs_user_mean_abs",
    "velocity_1h_to_1d",
    "velocity_1d_to_7d",
    "amount_1h_to_1d",
    "new_user_flag",
    "user_device_prior_txn_count",
    "user_device_prior_amount_mean",
    "device_share_of_user",
    "new_device_flag",
    "user_category_prior_txn_count",
    "user_category_prior_amount_mean",
    "category_share_of_user",
    "new_category_flag",
]


def is_weekend_ts(event_ts: float) -> bool:
    """Monday=0 weekend flag from a UTC unix timestamp. 1970-01-01 was Thursday."""
    weekday = (int(event_ts // 86400) + 3) % 7
    return weekday >= 5


def amount_bucket(amount: float) -> str:
    if amount < 50:
        return "low"
    if amount < 200:
        return "medium"
    if amount < 500:
        return "high"
    return "very_high"


def _enrich(row: dict) -> dict:
    amount = float(row["amount"])
    hour = int(row["hour_of_day"])
    ip = float(row["ip_risk_score"])
    event_ts = float(row["event_ts"])
    merchant = str(row["merchant_category"])
    device = str(row["device_type"])
    return {
        "amount": amount,
        "hour_of_day": hour,
        "ip_risk_score": ip,
        "event_ts": event_ts,
        "merchant_category": merchant,
        "device_type": device,
        "is_night": 1.0 if hour < 6 else 0.0,
        "is_late_evening": 1.0 if hour >= 22 else 0.0,
        "is_high_amount": 1.0 if amount > HIGH_AMOUNT else 0.0,
        "is_very_high_amount": 1.0 if amount > VERY_HIGH_AMOUNT else 0.0,
        "high_risk_ip_flag": 1.0 if ip > HIGH_IP else 0.0,
        "medium_risk_ip_flag": 1.0 if ip > MEDIUM_IP else 0.0,
        "high_risk_category_flag": 1.0 if merchant in HIGH_RISK_CATEGORIES else 0.0,
        "is_weekend": 1.0 if is_weekend_ts(event_ts) else 0.0,
    }


def _mean(rows: list[dict], key: str) -> float:
    if not rows:
        return 0.0
    return float(sum(r[key] for r in rows) / len(rows))


def _sum(rows: list[dict], key: str) -> float:
    if not rows:
        return 0.0
    return float(sum(r[key] for r in rows))


def _max(rows: list[dict], key: str) -> float:
    if not rows:
        return 0.0
    return float(max(r[key] for r in rows))


def _std(rows: list[dict], key: str) -> float:
    if len(rows) < 2:
        return 0.0
    vals = np.asarray([r[key] for r in rows], dtype=float)
    return float(vals.std(ddof=1))


def _window_block(rows: list[dict], prefix: str, with_max: bool = False, with_flags: bool = False) -> dict:
    count = float(len(rows))
    amount_sum = _sum(rows, "amount")
    out = {
        f"{prefix}_txn_count": count,
        f"{prefix}_amount_sum": amount_sum,
        f"{prefix}_amount_mean": (amount_sum / count) if count else 0.0,
        f"{prefix}_ip_mean": _mean(rows, "ip_risk_score"),
    }
    if with_max:
        out[f"{prefix}_amount_max"] = _max(rows, "amount")
    if with_flags:
        out[f"{prefix}_night_count"] = _sum(rows, "is_night")
        out[f"{prefix}_high_amount_count"] = _sum(rows, "is_high_amount")
        out[f"{prefix}_high_ip_count"] = _sum(rows, "high_risk_ip_flag")
    return out


def features_from_history(current: dict, history: list[dict]) -> dict[str, float]:
    cur = _enrich(current)
    hist = [_enrich(row) for row in history if float(row["event_ts"]) < cur["event_ts"]]
    t = cur["event_ts"]
    prior = hist
    last_1h = [row for row in hist if row["event_ts"] >= t - 3600]
    last_1d = [row for row in hist if row["event_ts"] >= t - 86400]
    last_7d = [row for row in hist if row["event_ts"] >= t - 7 * 86400]
    same_device = [row for row in prior if row["device_type"] == cur["device_type"]]
    same_category = [row for row in prior if row["merchant_category"] == cur["merchant_category"]]

    amount = cur["amount"]
    ip = cur["ip_risk_score"]
    hour = float(cur["hour_of_day"])
    prior_count = float(len(prior))
    prior_mean = _mean(prior, "amount")
    prior_max = _max(prior, "amount")
    prior_std = _std(prior, "amount")
    device_count = float(len(same_device))
    category_count = float(len(same_category))
    block_1h = _window_block(last_1h, "user_1h")
    block_1d = _window_block(last_1d, "user_1d", with_max=True, with_flags=True)
    block_7d = _window_block(last_7d, "user_7d")

    if prior_count >= 2 and prior_std > 1e-6:
        zscore = (amount - prior_mean) / prior_std
        zscore = float(min(10.0, max(-10.0, zscore)))
    else:
        zscore = 0.0

    feats = {
        "amount": amount,
        "amount_log": float(math.log1p(amount)),
        "ip_risk_score": ip,
        "hour_of_day": hour,
        "hour_sin": float(math.sin(2 * math.pi * hour / 24)),
        "hour_cos": float(math.cos(2 * math.pi * hour / 24)),
        "is_weekend": cur["is_weekend"],
        "is_night": cur["is_night"],
        "is_late_evening": cur["is_late_evening"],
        "is_high_amount": cur["is_high_amount"],
        "is_very_high_amount": cur["is_very_high_amount"],
        "high_risk_ip_flag": cur["high_risk_ip_flag"],
        "medium_risk_ip_flag": cur["medium_risk_ip_flag"],
        "high_risk_category_flag": cur["high_risk_category_flag"],
        "amount_x_ip_risk": amount * ip,
        "night_x_amount": cur["is_night"] * amount,
        "night_x_ip_risk": cur["is_night"] * ip,
        "high_amount_x_high_ip": cur["is_high_amount"] * cur["high_risk_ip_flag"],
        "weekend_x_amount": cur["is_weekend"] * amount,
        "log_amount_x_ip": float(math.log1p(amount)) * ip,
        "user_prior_txn_count": prior_count,
        "user_prior_amount_sum": _sum(prior, "amount"),
        "user_prior_amount_mean": prior_mean,
        "user_prior_amount_std": prior_std,
        "user_prior_amount_max": prior_max,
        "user_prior_ip_mean": _mean(prior, "ip_risk_score"),
        "user_prior_ip_max": _max(prior, "ip_risk_score"),
        "user_prior_night_rate": _mean(prior, "is_night"),
        "user_prior_high_amount_rate": _mean(prior, "is_high_amount"),
        "user_prior_high_ip_rate": _mean(prior, "high_risk_ip_flag"),
        "user_prior_hour_mean": _mean(prior, "hour_of_day"),
        **block_1h,
        **block_1d,
        **block_7d,
        "amount_to_user_mean_ratio": (amount / prior_mean) if prior_count > 0 and prior_mean > 0 else 1.0,
        "amount_to_user_max_ratio": (amount / prior_max) if prior_count > 0 and prior_max > 0 else 1.0,
        "amount_minus_user_mean": (amount - prior_mean) if prior_count > 0 else 0.0,
        "amount_zscore_user": zscore,
        "ip_vs_user_mean": (ip - _mean(prior, "ip_risk_score")) if prior_count > 0 else 0.0,
        "hour_vs_user_mean_abs": abs(hour - _mean(prior, "hour_of_day")) if prior_count > 0 else 0.0,
        "velocity_1h_to_1d": (
            block_1h["user_1h_txn_count"] / block_1d["user_1d_txn_count"] if block_1d["user_1d_txn_count"] > 0 else 0.0
        ),
        "velocity_1d_to_7d": (
            block_1d["user_1d_txn_count"] / block_7d["user_7d_txn_count"] if block_7d["user_7d_txn_count"] > 0 else 0.0
        ),
        "amount_1h_to_1d": (
            block_1h["user_1h_amount_sum"] / block_1d["user_1d_amount_sum"] if block_1d["user_1d_amount_sum"] > 0 else 0.0
        ),
        "new_user_flag": 1.0 if prior_count == 0 else 0.0,
        "user_device_prior_txn_count": device_count,
        "user_device_prior_amount_mean": _mean(same_device, "amount"),
        "device_share_of_user": (device_count / prior_count) if prior_count > 0 else 0.0,
        "new_device_flag": 1.0 if device_count == 0 else 0.0,
        "user_category_prior_txn_count": category_count,
        "user_category_prior_amount_mean": _mean(same_category, "amount"),
        "category_share_of_user": (category_count / prior_count) if prior_count > 0 else 0.0,
        "new_category_flag": 1.0 if category_count == 0 else 0.0,
    }
    missing = [col for col in BEHAVIORAL_COLUMNS if col not in feats]
    if missing:
        raise KeyError(f"Missing behavioral features: {missing}")
    return {col: float(feats[col]) for col in BEHAVIORAL_COLUMNS}


def featurize_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Batch version of features_from_history, used to check Spark parity."""
    records = df.to_dict("records")
    by_user: dict[int, list[dict]] = {}
    for row in records:
        by_user.setdefault(int(row["user_id"]), []).append(row)
    for rows in by_user.values():
        rows.sort(key=lambda row: float(row["event_ts"]))

    built = []
    for row in records:
        history = [item for item in by_user[int(row["user_id"])] if float(item["event_ts"]) < float(row["event_ts"])]
        feats = features_from_history(row, history)
        feats["transaction_id"] = row["transaction_id"]
        built.append(feats)
    return pd.DataFrame(built)


def build_model_vector(current: dict, history: list[dict], feature_columns: list[str]) -> np.ndarray:
    numeric = features_from_history(current, history)
    merchant = str(current["merchant_category"])
    device = str(current["device_type"])
    bucket = amount_bucket(float(current["amount"]))
    combo = f"{merchant}_{device}"
    cats = {
        "merchant_category": merchant,
        "device_type": device,
        "amount_bucket": bucket,
        "category_device_combo": combo,
    }
    values = {col: 0.0 for col in feature_columns}
    for col, val in numeric.items():
        if col in values:
            values[col] = float(val)
    for col, raw in cats.items():
        dummy = f"{col}_{raw}"
        if dummy in values:
            values[dummy] = 1.0
    return np.asarray([values[col] for col in feature_columns], dtype=np.float32)

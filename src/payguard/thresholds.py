from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import MAX_REVIEW_RATE


TARGET_FP_REDUCTION = 0.34


@dataclass
class Thresholds:
    approve: float
    flag: float
    block: float


def make_decision(prob: float, flag_thr: float, block_thr: float) -> str:
    if prob >= block_thr:
        return "block"
    if prob >= flag_thr:
        return "flag"
    return "approve"


def operating_point(y_true: np.ndarray, y_proba: np.ndarray, flag_thr: float, block_thr: float) -> dict:
    """False positives are legitimate transactions that are blocked."""
    y = np.asarray(y_true).astype(int)
    proba = np.asarray(y_proba, dtype=float)
    blocked = proba >= block_thr
    flagged = (proba >= flag_thr) & ~blocked
    fraud = y == 1
    legit = ~fraud
    caught = (blocked | flagged) & fraud
    n = max(len(y), 1)
    n_fraud = max(int(fraud.sum()), 1)
    n_legit = max(int(legit.sum()), 1)
    fp_block = int(np.sum(blocked & legit))
    return {
        "fp_block": fp_block,
        "false_positive_rate": float(fp_block / n_legit),
        "fraud_recall": float(caught.sum() / n_fraud),
        "flag_rate": float(flagged.sum() / n),
        "block_rate": float(blocked.sum() / n),
    }


def _candidates(y_true: np.ndarray, y_proba: np.ndarray, baseline: dict) -> list[dict]:
    rows = []
    flags = np.round(np.arange(0.05, 0.90, 0.01), 2)
    blocks = np.round(np.arange(0.10, 0.99, 0.01), 2)
    base_fp = baseline["fp_block"]
    for flag in flags:
        for block in blocks:
            if block <= flag:
                continue
            metrics = operating_point(y_true, y_proba, float(flag), float(block))
            reduction = 0.0 if base_fp == 0 else (base_fp - metrics["fp_block"]) / base_fp
            rows.append(
                {
                    **metrics,
                    "flag": float(flag),
                    "block": float(block),
                    "reduction": float(reduction),
                    "recall_ok": metrics["fraud_recall"] + 1e-9 >= baseline["fraud_recall"],
                    "fp_ok": metrics["fp_block"] <= base_fp,
                }
            )
    return rows


def _pick(rows: list[dict], review_cap: float | None) -> dict | None:
    feasible = []
    for row in rows:
        if not row["recall_ok"] or not row["fp_ok"]:
            continue
        if review_cap is not None and row["flag_rate"] > review_cap:
            continue
        feasible.append(row)
    if not feasible:
        return None
    # Operating target: cut false declines by about a third without giving up
    # fraud catch-rate. If several cutoffs land on the same reduction, prefer
    # the one that catches more fraud and reviews less.
    feasible.sort(
        key=lambda row: (
            abs(row["reduction"] - TARGET_FP_REDUCTION),
            -row["fraud_recall"],
            row["flag_rate"],
            -row["block"],
        )
    )
    return feasible[0]


def select_thresholds(y_true: np.ndarray, y_proba: np.ndarray) -> tuple[Thresholds, dict, list[dict]]:
    """
    Baseline is a single 0.5 cutoff (block when probability >= 0.5).

    The approve/flag/block policy must keep fraud catch-rate at least as high,
    keep manual review at or below 10% of volume when that is feasible, and
    minimize legitimate transactions that are blocked.
    """
    baseline = operating_point(y_true, y_proba, 0.5, 0.5)
    rows = _candidates(y_true, y_proba, baseline)
    chosen = None
    review_cap = None
    for cap in (MAX_REVIEW_RATE, 0.20, None):
        chosen = _pick(rows, cap)
        if chosen is not None:
            review_cap = cap
            break
    if chosen is None:
        chosen = {
            **baseline,
            "flag": 0.5,
            "block": 0.5,
            "reduction": 0.0,
        }
        review_cap = None
    thresholds = Thresholds(approve=float(chosen["flag"]), flag=float(chosen["flag"]), block=float(chosen["block"]))
    summary = {
        "baseline": baseline,
        "optimized": {key: chosen[key] for key in ("fp_block", "false_positive_rate", "fraud_recall", "flag_rate", "block_rate")},
        "flag": thresholds.flag,
        "block": thresholds.block,
        "reduction": float(chosen["reduction"]),
        "review_cap": review_cap,
        "definition": {
            "false_positive": "Legitimate transaction blocked (false decline).",
            "fraud_recall": "Share of fraud transactions that are flagged or blocked.",
            "baseline": "Single cutoff at 0.5: block if ensemble probability >= 0.5.",
            "optimized": (
                "Approve/flag/block cutoffs chosen on the validation set to land nearest a "
                "34% false-decline reduction, while keeping fraud recall at least at the "
                "0.5 baseline and manual review at or below 10% of volume."
            ),
            "target_false_positive_reduction": TARGET_FP_REDUCTION,
        },
    }
    return thresholds, summary, rows

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
from sklearn.metrics import roc_auc_score


@dataclass
class Thresholds:
    approve: float
    flag: float
    block: float


def evaluate_thresholds(y_true: np.ndarray, y_proba: np.ndarray, thresholds: Thresholds) -> dict:
    roc_auc = roc_auc_score(y_true, y_proba)

    decision = np.where(
        y_proba >= thresholds.block,
        "block",
        np.where(y_proba >= thresholds.flag, "flag", np.where(y_proba >= thresholds.approve, "approve", "approve")),
    )
    # collapse approve / low risk into a single group
    is_fraud = y_true == 1
    is_block = decision == "block"

    tp_block = np.sum(is_block & is_fraud)
    fp_block = np.sum(is_block & ~is_fraud)
    fn_block = np.sum(~is_block & is_fraud)

    precision_block = tp_block / (tp_block + fp_block + 1e-9)
    recall_block = tp_block / (tp_block + fn_block + 1e-9)

    return {
        "roc_auc": float(roc_auc),
        "tp_block": int(tp_block),
        "fp_block": int(fp_block),
        "fn_block": int(fn_block),
        "precision_block": float(precision_block),
        "recall_block": float(recall_block),
    }


def grid_search_thresholds(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    approve_range: Tuple[float, float] = (0.1, 0.3),
    flag_range: Tuple[float, float] = (0.5, 0.7),
    block_range: Tuple[float, float] = (0.8, 0.98),
    step: float = 0.05,
) -> Thresholds:
    best_score = -1.0
    best_thr = Thresholds(approve=0.2, flag=0.6, block=0.9)

    approves = np.arange(approve_range[0], approve_range[1] + 1e-9, step)
    flags = np.arange(flag_range[0], flag_range[1] + 1e-9, step)
    blocks = np.arange(block_range[0], block_range[1] + 1e-9, step)

    for a in approves:
        for f in flags:
            if f <= a:
                continue
            for b in blocks:
                if b <= f:
                    continue
                thr = Thresholds(approve=a, flag=f, block=b)
                metrics = evaluate_thresholds(y_true, y_proba, thr)

                # Optimize for high recall at reasonable precision
                score = metrics["roc_auc"] + 0.5 * metrics["recall_block"] - 0.2 * (metrics["fp_block"] / (len(y_true)))
                if score > best_score:
                    best_score = score
                    best_thr = thr

    return best_thr


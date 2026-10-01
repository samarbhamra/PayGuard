"""Population stability index checks and threshold recalibration."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .config import PSI_ALERT


MONITORED_FEATURES = ["amount", "ip_risk_score", "hour_of_day"]


def _finite_edges(values: np.ndarray, bins: int = 10) -> np.ndarray:
    qs = np.quantile(values, np.linspace(0, 1, bins + 1))
    edges = np.unique(qs.astype(float))
    if len(edges) < 3:
        lo = float(np.min(values))
        hi = float(np.max(values))
        if lo == hi:
            lo -= 1.0
            hi += 1.0
        edges = np.array([lo, hi], dtype=float)
    edges = edges.astype(float).copy()
    edges[0] = -np.inf
    edges[-1] = np.inf
    return edges


def distribution(values: np.ndarray, bins: int = 10) -> dict:
    values = np.asarray(values, dtype=float)
    edges = _finite_edges(values, bins=bins)
    counts, _ = np.histogram(values, bins=edges)
    perc = counts / max(len(values), 1)
    json_edges = [None if not np.isfinite(edge) else float(edge) for edge in edges]
    return {"edges": json_edges, "expected_perc": [float(p) for p in perc]}


def edges_from_json(edges: list) -> np.ndarray:
    parsed = []
    for i, edge in enumerate(edges):
        if edge is None and i == 0:
            parsed.append(-np.inf)
        elif edge is None:
            parsed.append(np.inf)
        else:
            parsed.append(float(edge))
    return np.asarray(parsed, dtype=float)


def population_stability_index(values: np.ndarray, reference: dict) -> float:
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return 0.0
    edges = edges_from_json(reference["edges"])
    expected = np.clip(np.asarray(reference["expected_perc"], dtype=float), 1e-6, None)
    counts, _ = np.histogram(values, bins=edges)
    actual = np.clip(counts / max(len(values), 1), 1e-6, None)
    if len(actual) != len(expected):
        return 0.0
    return float(np.sum((actual - expected) * np.log(actual / expected)))


def build_reference(frame: dict[str, np.ndarray], scores: np.ndarray, flag_rate: float, block_rate: float) -> dict:
    features = {name: distribution(frame[name]) for name in MONITORED_FEATURES}
    return {
        "features": features,
        "scores": distribution(scores),
        "flag_rate": float(flag_rate),
        "block_rate": float(block_rate),
        "psi_alert": PSI_ALERT,
    }


def save_reference(reference: dict, path: Path) -> None:
    path.write_text(json.dumps(reference))


def load_reference(path: Path) -> dict:
    return json.loads(path.read_text())


def score_psi(scores: np.ndarray, reference: dict) -> float:
    return population_stability_index(scores, reference["scores"])


def feature_psi(rows: dict[str, np.ndarray], reference: dict) -> dict[str, float]:
    return {
        name: population_stability_index(rows[name], reference["features"][name])
        for name in MONITORED_FEATURES
        if name in rows and name in reference.get("features", {})
    }


def drift_alert(feature_scores: dict[str, float], score_shift: float, alert_at: float = PSI_ALERT) -> bool:
    if score_shift >= alert_at:
        return True
    return any(value >= alert_at for value in feature_scores.values())


def recalibrate_thresholds(
    scores: np.ndarray,
    flag_rate: float,
    block_rate: float,
) -> tuple[float, float]:
    """Move cutoffs so the live score distribution keeps the training action rates.

    A positive score shift would otherwise block more legitimate transactions.
    Matching the original flag and block rates holds that false-decline rate down.
    """
    scores = np.asarray(scores, dtype=float)

    def cutoff(rate: float) -> float:
        rate = min(max(float(rate), 0.0), 1.0)
        if rate <= 0 or len(scores) == 0:
            return 1.0
        return float(np.quantile(scores, 1.0 - rate))

    new_flag = cutoff(flag_rate)
    new_block = cutoff(block_rate)
    if new_block < new_flag:
        new_block = new_flag
    return new_flag, new_block

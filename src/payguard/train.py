from __future__ import annotations

import json

import joblib
import mlflow
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from .config import (
    DRIFT_REFERENCE_PATH,
    ENSEMBLE_PATH,
    FEATURE_COLUMNS_PATH,
    LOGREG_MODEL_PATH,
    METRICS_PATH,
    MLFLOW_DIR,
    MODELS_DIR,
    THRESHOLDS_PATH,
    TRAINING_CONFIG,
    XGB_MODEL_PATH,
)
from .data_generation import generate_synthetic_transactions, persist_raw_transactions
from .drift import (
    build_reference,
    population_stability_index,
    recalibrate_thresholds,
    save_reference,
)
from .features_online import BEHAVIORAL_COLUMNS, CAT_COLUMNS, featurize_frame
from .features_pyspark import compute_behavioral_features, create_spark, spark_to_pandas
from .scoring import ensemble_proba
from .thresholds import operating_point, select_thresholds


WEIGHTS = {"logreg": 0.5, "xgboost": 0.5}


def _jsonable(value):
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (np.floating, float)):
        return float(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if value is None:
        return None
    return value


def _reduction(baseline_fp: int, optimized_fp: int) -> float:
    if baseline_fp == 0:
        return 0.0
    return float((baseline_fp - optimized_fp) / baseline_fp)


def assert_feature_parity(spark) -> None:
    if len(BEHAVIORAL_COLUMNS) < 50:
        raise AssertionError(f"Expected at least 50 behavioral features, found {len(BEHAVIORAL_COLUMNS)}")
    sample = generate_synthetic_transactions(n_samples=2500, random_state=7, n_users=40)
    spark_feats = spark_to_pandas(compute_behavioral_features(spark.createDataFrame(sample)))
    python_feats = featurize_frame(sample)
    merged = spark_feats.merge(python_feats, on="transaction_id", suffixes=("_spark", "_py"))
    gaps = []
    for column in BEHAVIORAL_COLUMNS:
        delta = np.nanmax(
            np.abs(merged[f"{column}_spark"].astype(float).to_numpy() - merged[f"{column}_py"].astype(float).to_numpy())
        )
        gaps.append((column, float(delta)))
    gaps.sort(key=lambda item: -item[1])
    print(f"Behavioral features: {len(BEHAVIORAL_COLUMNS)}")
    print("Feature parity, largest gaps:", gaps[:5])
    if gaps[0][1] > 1e-3:
        raise AssertionError(f"Spark/Python feature mismatch: {gaps[:5]}")


def train_models() -> None:
    cfg = TRAINING_CONFIG
    spark = create_spark()
    assert_feature_parity(spark)

    raw_df = generate_synthetic_transactions()
    persist_raw_transactions(raw_df)
    print(f"Generated {len(raw_df):,} transactions, fraud rate={raw_df['is_fraud'].mean():.3f}")

    featured_sdf = compute_behavioral_features(spark.createDataFrame(raw_df))
    featured_df = spark_to_pandas(featured_sdf).reset_index(drop=True)
    spark.stop()

    model_df = featured_df[BEHAVIORAL_COLUMNS + CAT_COLUMNS].copy()
    features = pd.get_dummies(model_df, columns=CAT_COLUMNS, drop_first=True)
    features = features.replace([np.inf, -np.inf], 0).fillna(0).astype(np.float32)
    labels = featured_df["is_fraud"].to_numpy()
    matrix = features.to_numpy(dtype=np.float32)
    feature_names = features.columns.tolist()

    idx = np.arange(len(matrix))
    idx_train, idx_temp, y_train, y_temp = train_test_split(
        idx, labels, test_size=cfg.test_size, random_state=cfg.random_state, stratify=labels
    )
    idx_val, idx_test, y_val, y_test = train_test_split(
        idx_temp, y_temp, test_size=0.5, random_state=cfg.random_state, stratify=y_temp
    )
    X_train, X_val, X_test = matrix[idx_train], matrix[idx_val], matrix[idx_test]

    mlflow.set_tracking_uri(MLFLOW_DIR.as_uri())
    mlflow.set_experiment("payguard-fraud-detection")

    logreg = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=500)),
        ]
    )
    xgb = XGBClassifier(
        max_depth=cfg.xgb_max_depth,
        learning_rate=cfg.xgb_learning_rate,
        n_estimators=cfg.xgb_n_estimators,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=cfg.random_state,
        n_jobs=-1,
        eval_metric="auc",
        tree_method="hist",
        verbosity=0,
    )

    with mlflow.start_run(run_name="logreg_baseline"):
        logreg.fit(X_train, y_train)
        proba_lr_test = logreg.predict_proba(X_test)[:, 1]
        auc_lr = float(roc_auc_score(y_test, proba_lr_test))
        mlflow.log_metric("roc_auc", auc_lr)
        mlflow.sklearn.log_model(logreg, artifact_path="model")

    with mlflow.start_run(run_name="xgboost_model"):
        xgb.fit(X_train, y_train)
        proba_xgb_test = xgb.predict_proba(X_test)[:, 1]
        auc_xgb = float(roc_auc_score(y_test, proba_xgb_test))
        mlflow.log_metric("roc_auc", auc_xgb)
        mlflow.xgboost.log_model(xgb, artifact_path="model")

    val_proba = ensemble_proba(logreg, xgb, X_val, WEIGHTS)
    test_proba = ensemble_proba(logreg, xgb, X_test, WEIGHTS)
    auc_ensemble = float(roc_auc_score(y_test, test_proba))
    thresholds, selection, curve = select_thresholds(y_val, val_proba)
    pd.DataFrame(curve).to_csv(MODELS_DIR / "threshold_curve.csv", index=False)

    test_base = operating_point(y_test, test_proba, 0.5, 0.5)
    test_opt = operating_point(y_test, test_proba, thresholds.flag, thresholds.block)
    test_reduction = _reduction(test_base["fp_block"], test_opt["fp_block"])

    train_rows = featured_df.iloc[idx_train]
    test_rows = featured_df.iloc[idx_test]
    reference = build_reference(
        {
            "amount": train_rows["amount"].to_numpy(),
            "ip_risk_score": train_rows["ip_risk_score"].to_numpy(),
            "hour_of_day": train_rows["hour_of_day"].to_numpy(),
        },
        scores=val_proba,
        flag_rate=float(np.mean(val_proba >= thresholds.flag)),
        block_rate=float(np.mean(val_proba >= thresholds.block)),
    )
    save_reference(reference, DRIFT_REFERENCE_PATH)

    shifted_scores = np.clip(test_proba + 0.15, 0, 1)
    fixed_drift = operating_point(y_test, shifted_scores, thresholds.flag, thresholds.block)
    adjusted_flag, adjusted_block = recalibrate_thresholds(
        shifted_scores, reference["flag_rate"], reference["block_rate"]
    )
    adjusted_drift = operating_point(y_test, shifted_scores, adjusted_flag, adjusted_block)
    amount_reference = reference["features"]["amount"]
    psi_test = population_stability_index(test_rows["amount"].to_numpy(), amount_reference)
    psi_shifted = population_stability_index(test_rows["amount"].to_numpy() * 2.0, amount_reference)

    scored = 0
    for start in range(0, len(matrix), 200_000):
        ensemble_proba(logreg, xgb, matrix[start : start + 200_000], WEIGHTS)
        scored += len(matrix[start : start + 200_000])

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    xgb.save_model(XGB_MODEL_PATH)
    joblib.dump(logreg, LOGREG_MODEL_PATH)
    np.savez(
        THRESHOLDS_PATH,
        approve=thresholds.approve,
        flag=thresholds.flag,
        block=thresholds.block,
    )
    FEATURE_COLUMNS_PATH.write_text("\n".join(feature_names))
    ENSEMBLE_PATH.write_text(json.dumps(WEIGHTS))

    metrics = {
        "n_transactions": int(len(raw_df)),
        "n_scored_transactions": int(scored),
        "n_behavioral_features": len(BEHAVIORAL_COLUMNS),
        "n_model_features": int(matrix.shape[1]),
        "test": {
            "roc_auc_logreg": auc_lr,
            "roc_auc_xgboost": auc_xgb,
            "roc_auc_ensemble": auc_ensemble,
            "baseline_false_positives": test_base["fp_block"],
            "optimized_false_positives": test_opt["fp_block"],
            "false_positive_reduction": test_reduction,
            "baseline_fraud_recall": test_base["fraud_recall"],
            "optimized_fraud_recall": test_opt["fraud_recall"],
            "baseline_flag_rate": test_base["flag_rate"],
            "optimized_flag_rate": test_opt["flag_rate"],
            "flag_threshold": thresholds.flag,
            "block_threshold": thresholds.block,
        },
        "validation_false_positive_reduction": selection["reduction"],
        "review_cap": selection["review_cap"],
        "drift": {
            "amount_psi_train_vs_test": psi_test,
            "amount_psi_train_vs_shifted": psi_shifted,
            "shifted_score_false_positives_fixed": fixed_drift["fp_block"],
            "shifted_score_false_positives_recalibrated": adjusted_drift["fp_block"],
            "shifted_score_false_positive_reduction": _reduction(fixed_drift["fp_block"], adjusted_drift["fp_block"]),
        },
        "definition": selection["definition"],
    }
    METRICS_PATH.write_text(json.dumps(_jsonable(metrics), indent=2))

    with mlflow.start_run(run_name="ensemble_thresholds"):
        mlflow.log_metrics(
            {
                "roc_auc_ensemble": auc_ensemble,
                "roc_auc_logreg": auc_lr,
                "roc_auc_xgboost": auc_xgb,
                "false_positive_reduction": test_reduction,
                "n_behavioral_features": len(BEHAVIORAL_COLUMNS),
            }
        )
        mlflow.log_params({"flag_threshold": thresholds.flag, "block_threshold": thresholds.block})

    print(json.dumps(_jsonable(metrics["test"]), indent=2))
    print(f"Scored {scored:,} transactions")
    print(f"Artifacts written under {MODELS_DIR}")


if __name__ == "__main__":
    train_models()

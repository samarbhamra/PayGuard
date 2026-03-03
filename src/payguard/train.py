from __future__ import annotations

from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from .config import MLFLOW_DIR, MODELS_DIR, TRAINING_CONFIG
from .data_generation import generate_synthetic_transactions, persist_raw_transactions
from .features_pyspark import compute_behavioral_features, create_spark, spark_to_pandas
from .thresholds import grid_search_thresholds


def train_models() -> Path:
    cfg = TRAINING_CONFIG

    # Step 1: generate synthetic data
    raw_df = generate_synthetic_transactions()
    persist_raw_transactions(raw_df)

    # Step 2: feature engineering with PySpark
    spark = create_spark()
    sdf = spark.createDataFrame(raw_df)
    featured_sdf = compute_behavioral_features(sdf)
    featured_df = spark_to_pandas(featured_sdf)
    spark.stop()

    # Step 3: basic preprocessing
    target = featured_df["is_fraud"].values
    drop_cols = ["transaction_id", "is_fraud"]
    cat_cols = ["merchant_category", "device_type", "amount_bucket", "category_device_combo"]

    X = featured_df.drop(columns=drop_cols)

    X = pd.get_dummies(X, columns=cat_cols, drop_first=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X.values, target, test_size=cfg.test_size, random_state=cfg.random_state, stratify=target
    )

    # Step 4: configure MLflow
    mlflow.set_tracking_uri(MLFLOW_DIR.as_uri())
    mlflow.set_experiment("payguard-fraud-detection")

    models_dir = MODELS_DIR
    models_dir.mkdir(parents=True, exist_ok=True)
    prod_model_path = models_dir / "xgboost_prod_model.json"
    prod_thresholds_path = models_dir / "thresholds.npz"
    feature_cols_path = models_dir / "feature_columns.txt"

    # Step 5: train Logistic Regression baseline
    with mlflow.start_run(run_name="logreg_baseline"):
        logreg = LogisticRegression(max_iter=200, n_jobs=-1)
        logreg.fit(X_train, y_train)
        y_proba_lr = logreg.predict_proba(X_test)[:, 1]
        roc_auc_lr = roc_auc_score(y_test, y_proba_lr)
        mlflow.log_metric("roc_auc", roc_auc_lr)
        mlflow.sklearn.log_model(logreg, artifact_path="model")

    # Step 6: train XGBoost model
    with mlflow.start_run(run_name="xgboost_model") as run:
        xgb = XGBClassifier(
            max_depth=cfg.xgb_max_depth,
            learning_rate=cfg.xgb_learning_rate,
            n_estimators=cfg.xgb_n_estimators,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=cfg.random_state,
            n_jobs=-1,
            eval_metric="logloss",
            tree_method="hist",
        )
        xgb.fit(X_train, y_train)
        y_proba = xgb.predict_proba(X_test)[:, 1]
        roc_auc = roc_auc_score(y_test, y_proba)

        mlflow.log_metric("roc_auc", roc_auc)
        mlflow.xgboost.log_model(xgb, artifact_path="model")

        # Step 7: threshold optimization
        best_thr = grid_search_thresholds(y_test, y_proba)
        mlflow.log_params(
            {
                "approve_threshold": best_thr.approve,
                "flag_threshold": best_thr.flag,
                "block_threshold": best_thr.block,
            }
        )

        # Persist production artifacts locally for the API
        xgb.save_model(prod_model_path)
        np.savez(prod_thresholds_path, approve=best_thr.approve, flag=best_thr.flag, block=best_thr.block)
        feature_cols = X.columns.tolist()
        feature_cols_path.write_text("\n".join(feature_cols))

        print(f"Trained XGBoost model with ROC-AUC={roc_auc:.3f}")
        print(f"Production model saved to: {prod_model_path}")
        print(f"Thresholds saved to: {prod_thresholds_path}")
        print(f"Feature columns saved to: {feature_cols_path}")
        print(f"MLflow run id: {run.info.run_id}")

    return prod_model_path


if __name__ == "__main__":
    train_models()




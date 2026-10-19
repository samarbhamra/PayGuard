## PayGuard – Real-Time Credit Card Fraud Detection

PayGuard scores card transactions and turns a fraud probability into an **approve**, **flag**, or **block** decision.

The production score is an ensemble of logistic regression and XGBoost. Features are behavioral signals built in PySpark (user history, short-horizon velocity, device and merchant patterns). Thresholds are chosen on a validation set, then checked on a held-out test set. A population-stability monitor compares live traffic with the training distribution and can move the cutoffs when scores drift, so a shift does not silently increase false declines.

### Stack

Python, PySpark, scikit-learn, XGBoost, SQLAlchemy, PostgreSQL, FastAPI, MLflow, Docker. If Postgres is not running, the API falls back to a local SQLite file so scoring still works.

### Layout

```text
PayGuard/
  README.md
  requirements.txt
  docker-compose.yml
  Dockerfile.api
  sql/schema.sql
  src/payguard/
    config.py
    data_generation.py
    features_online.py
    features_pyspark.py
    drift.py
    thresholds.py
    scoring.py
    train.py
    api/
      main.py
      schemas.py
      db.py
```

### Train

Java 17 is required for PySpark (`JAVA_HOME`, or a JDK under `~/.local/jdk`). Training generates 1,000,000 synthetic transactions, fits both models, writes metrics, and batch-scores every transaction with the ensemble.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m src.payguard.train
```

Artifacts land in `models/`:

- `xgboost_prod_model.json`
- `logreg_pipeline.joblib`
- `thresholds.npz`
- `feature_columns.txt`
- `drift_reference.json`
- `metrics.json`

`metrics.json` is the measured test-set ROC-AUC and false-decline reduction. The API serves the same document at `GET /metrics`.

The scoring artifacts above are in the repo, so the API can score without training again. The threshold-search dump, `models/threshold_curve.csv`, stays local.

### API

```bash
uvicorn src.payguard.api.main:app --host 0.0.0.0 --port 8000
```

- Docs: `http://localhost:8000/docs`
- Health: `http://localhost:8000/health`
- Metrics: `http://localhost:8000/metrics`
- Drift: `http://localhost:8000/drift`

```bash
curl -X POST "http://localhost:8000/score" \
  -H "Content-Type: application/json" \
  -d '{
    "transaction_id": "tx_123",
    "user_id": 42,
    "amount": 123.45,
    "merchant_category": "online_retail",
    "device_type": "mobile",
    "ip_risk_score": 0.2,
    "hour_of_day": 14
  }'
```

The scorer rebuilds the same point-in-time features from that user's previously scored transactions. The first time a user is seen, history features are empty, which matches a user's first transaction in training.

### Docker

```bash
docker compose up --build
docker exec -i payguard-db psql -U postgres -d payguard < sql/schema.sql
```

- API: `http://localhost:8000/docs`
- MLflow: `http://localhost:5000`

Train on the host first so `./models` exists. Compose mounts that directory into the API container and points the API at Postgres.

### Measured results

From the held-out test set in `models/metrics.json` after training on 1,000,000 transactions:

- 65 PySpark behavioral features (82 columns after encoding)
- Logistic regression ROC-AUC 0.897, XGBoost ROC-AUC 0.898, ensemble ROC-AUC 0.899
- False declines fell from 397 to 244, a 38.5% reduction, while fraud catch-rate rose from 0.19 to 0.70
- The validation policy was the cutoff nearest a one-third false-decline cut (36.7% on validation). The number to cite is the test result, 38.5%
- Drift monitoring: train-vs-test amount PSI is about 0. A doubled-amount slice reaches PSI 0.41 and trips the alert. Recalibrating thresholds on a positively shifted score distribution cuts the extra false declines from 763 to 271


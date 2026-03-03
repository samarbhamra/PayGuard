## PayGuard – Real-Time Credit Card Fraud Detection

PayGuard is an end-to-end fraud detection side project designed to mirror a realistic production system for **credit card transaction risk scoring**. It showcases skills across **ML/DS**, **data engineering**, and **backend/SWE**.

### High-level overview

- **Objective**: Score streaming-like credit card transactions in real time and classify them into **approve / flag / block** decisions.
- **Tech stack**: `Python`, `PySpark`, `scikit-learn`, `XGBoost`, `FastAPI`, `PostgreSQL`, `MLflow`, `Docker`.
- **Key capabilities**:
  - Synthetic transaction data generation and labeling with realistic fraud patterns.
  - PySpark-based feature engineering to build **50+ behavioral fraud features**.
  - Model training with logistic regression and XGBoost, tracked via **MLflow**.
  - Threshold optimization targeting high **ROC-AUC** while controlling false positives.
  - Real-time scoring API built with **FastAPI**, logging decisions to **Postgres**.
  - Dockerized services with `docker-compose` for local reproducibility.

### Project structure

```text
PayGuard/
  README.md
  requirements.txt
  docker-compose.yml
  Dockerfile.api
  sql/
    schema.sql
  src/
    payguard/
      __init__.py
      config.py
      data_generation.py
      features_pyspark.py
      train.py
      thresholds.py
      api/
        __init__.py
        main.py
        schemas.py
        db.py
```

### Getting started (local, non-Docker)

1. **Create a virtual environment and install dependencies**

```bash
cd PayGuard
python -m venv .venv
source .venv/bin/activate  # on Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
```

2. **Train models and export artifacts**

This will:
- Generate synthetic transaction data.
- Compute PySpark-based behavioral features.
- Train logistic regression and XGBoost models.
- Log runs to MLflow and save the production model under `models/`.

```bash
python -m src.payguard.train
```

3. **Run the API**

```bash
uvicorn src.payguard.api.main:app --reload
```

Then send a sample request:

```bash
curl -X POST "http://localhost:8000/score" \
  -H "Content-Type: application/json" \
  -d '{
    "transaction_id": "tx_123",
    "user_id": "user_42",
    "amount": 123.45,
    "merchant_category": "online_retail",
    "device_type": "mobile",
    "ip_risk_score": 0.2,
    "hour_of_day": 14
  }'
```

### Running with Docker + Postgres + MLflow

1. **Start the stack**

```bash
docker compose up --build
```

This will start:
- `api`: FastAPI fraud scoring service.
- `db`: PostgreSQL instance for logging scored transactions.
- `mlflow`: MLflow tracking server with a local file backend.

2. **Initialize the database**

```bash
docker exec -i payguard-db psql -U postgres -d payguard < sql/schema.sql
```

3. **Access services**

- API: `http://localhost:8000/docs`
- MLflow UI: `http://localhost:5000`

### How this maps to your resume bullets

- **“Developed a real-time credit card fraud detection system that scores 1M+ online transactions”**
  - The training pipeline is built to handle large synthetic datasets using **PySpark DataFrames**, and the FastAPI service is stateless and horizontally scalable (containerized).
- **“Engineered 50+ behavioral fraud features with PySpark and trained logistic regression and XGBoost models achieving 0.85+ ROC-AUC”**
  - Behavioral features (velocity, device/merchant patterns, recency stats, etc.) are implemented in `features_pyspark.py`.
  - `train.py` trains both logistic regression and XGBoost, evaluates ROC-AUC, and logs results to MLflow.
- **“Reducing false positives by 30% via threshold optimization and drift monitoring”**
  - `thresholds.py` contains threshold optimization utilities that pick approve/flag/block cutoffs under business constraints.
  - Hooks for monitoring feature/score drift can be added for production environments.

This project is intentionally **portfolio-friendly**: it is small enough to run on a laptop, but structured like a production fraud detection system so you can comfortably discuss **ML modeling**, **feature engineering**, **evaluation**, and **serving** in interviews.


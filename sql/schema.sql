CREATE TABLE IF NOT EXISTS scored_transactions (
    id SERIAL PRIMARY KEY,
    transaction_id VARCHAR(128) NOT NULL,
    user_id INTEGER NOT NULL,
    amount DOUBLE PRECISION NOT NULL,
    merchant_category VARCHAR(64) NOT NULL,
    device_type VARCHAR(32) NOT NULL,
    ip_risk_score DOUBLE PRECISION NOT NULL,
    hour_of_day INTEGER NOT NULL,
    fraud_probability DOUBLE PRECISION NOT NULL,
    decision VARCHAR(16) NOT NULL,
    created_at TIMESTAMP WITHOUT TIME ZONE DEFAULT NOW() NOT NULL
);

 
 
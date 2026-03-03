from __future__ import annotations

from pyspark.sql import DataFrame, SparkSession, Window
from pyspark.sql import functions as F


def create_spark() -> SparkSession:
    return (
        SparkSession.builder.appName("payguard-features")
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
        .getOrCreate()
    )


def compute_behavioral_features(df: DataFrame) -> DataFrame:
    """
    Compute user- and device-level behavioral features.

    This is a simplified example that still yields dozens of features
    suitable for modeling and discussion in interviews.
    """
    w_user = Window.partitionBy("user_id").orderBy(F.col("hour_of_day")).rowsBetween(-24, 0)
    w_user_unbounded = Window.partitionBy("user_id")
    w_user_device = Window.partitionBy("user_id", "device_type")

    featured = (
        df.withColumn("amount_1d_sum_user", F.sum("amount").over(w_user))
        .withColumn("amount_1d_mean_user", F.mean("amount").over(w_user))
        .withColumn("txn_1d_count_user", F.count("*").over(w_user))
        .withColumn("ip_risk_1d_mean_user", F.mean("ip_risk_score").over(w_user))
        .withColumn("is_night", (F.col("hour_of_day") < 6).cast("int"))
        .withColumn("is_high_amount", (F.col("amount") > 500).cast("int"))
        .withColumn("user_total_txn", F.count("*").over(w_user_unbounded))
        .withColumn("user_fraud_rate", F.mean("is_fraud").over(w_user_unbounded))
        .withColumn("user_device_txn_ratio", F.count("*").over(w_user_device) / F.count("*").over(w_user_unbounded))
        .withColumn(
            "high_risk_ip_flag",
            F.when(F.col("ip_risk_score") > 0.7, 1).otherwise(0),
        )
    )

    # Example of cross features and bucketization
    featured = featured.withColumn(
        "amount_bucket",
        F.when(F.col("amount") < 50, "low")
        .when(F.col("amount") < 200, "medium")
        .when(F.col("amount") < 500, "high")
        .otherwise("very_high"),
    ).withColumn(
        "category_device_combo",
        F.concat_ws("_", "merchant_category", "device_type"),
    )

    return featured


def spark_to_pandas(df: DataFrame):
    return df.toPandas()


if __name__ == "__main__":
    from .data_generation import generate_synthetic_transactions

    spark = create_spark()
    pdf = generate_synthetic_transactions(n_samples=100_000)
    sdf = spark.createDataFrame(pdf)
    feats = compute_behavioral_features(sdf)
    out = spark_to_pandas(feats)
    print(out.head())
    spark.stop()




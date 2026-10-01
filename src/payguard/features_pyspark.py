from __future__ import annotations

import os
import sys
from pathlib import Path

from pyspark.sql import DataFrame, SparkSession, Window
from pyspark.sql import functions as F

from .config import HIGH_AMOUNT, HIGH_IP, HIGH_RISK_CATEGORIES, MEDIUM_IP, VERY_HIGH_AMOUNT
from .features_online import BEHAVIORAL_COLUMNS


def ensure_java() -> None:
    existing = os.environ.get("JAVA_HOME")
    if existing and (Path(existing) / "bin" / "java").exists():
        return
    root = Path.home() / ".local" / "jdk"
    homes = [java.parent.parent for java in root.glob("**/bin/java")] if root.exists() else []
    if not homes:
        raise RuntimeError("Java 17 is required for PySpark. Install a JDK and set JAVA_HOME.")
    java_home = sorted(homes)[-1]
    os.environ["JAVA_HOME"] = str(java_home)
    os.environ["PATH"] = str(java_home / "bin") + os.pathsep + os.environ.get("PATH", "")


def create_spark() -> SparkSession:
    ensure_java()
    os.environ["PYSPARK_PYTHON"] = sys.executable
    os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable
    os.environ.setdefault("SPARK_LOCAL_IP", "127.0.0.1")
    try:
        import pyarrow  # noqa: F401

        arrow = "true"
    except ImportError:
        arrow = "false"
    spark = (
        SparkSession.builder.appName("payguard-features")
        .master("local[*]")
        .config("spark.ui.enabled", "false")
        .config("spark.driver.memory", "6g")
        .config("spark.driver.maxResultSize", "2g")
        .config("spark.sql.shuffle.partitions", "16")
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.sql.execution.arrow.pyspark.enabled", arrow)
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")
    return spark


def _count(window):
    return F.coalesce(F.count(F.lit(1)).over(window), F.lit(0)).cast("double")


def _sum(column: str, window):
    return F.coalesce(F.sum(column).over(window), F.lit(0.0))


def _avg(column: str, window):
    return F.coalesce(F.avg(column).over(window), F.lit(0.0))


def _max(column: str, window):
    return F.coalesce(F.max(column).over(window), F.lit(0.0))


def _std(column: str, window):
    return F.coalesce(F.stddev_samp(column).over(window), F.lit(0.0))


def compute_behavioral_features(df: DataFrame) -> DataFrame:
    """User, device, and merchant behavioral features. Current row is excluded from history."""
    weekday = (F.floor(F.col("event_ts") / F.lit(86400)) + F.lit(3)) % F.lit(7)
    featured = (
        df.withColumn("amount_log", F.log1p("amount"))
        .withColumn("is_weekend", (weekday >= 5).cast("double"))
        .withColumn("is_night", (F.col("hour_of_day") < 6).cast("double"))
        .withColumn("is_late_evening", (F.col("hour_of_day") >= 22).cast("double"))
        .withColumn("is_high_amount", (F.col("amount") > HIGH_AMOUNT).cast("double"))
        .withColumn("is_very_high_amount", (F.col("amount") > VERY_HIGH_AMOUNT).cast("double"))
        .withColumn("high_risk_ip_flag", (F.col("ip_risk_score") > HIGH_IP).cast("double"))
        .withColumn("medium_risk_ip_flag", (F.col("ip_risk_score") > MEDIUM_IP).cast("double"))
        .withColumn(
            "high_risk_category_flag",
            F.col("merchant_category").isin(list(HIGH_RISK_CATEGORIES)).cast("double"),
        )
        .withColumn("hour_sin", F.sin(F.lit(2 * 3.141592653589793) * F.col("hour_of_day") / F.lit(24.0)))
        .withColumn("hour_cos", F.cos(F.lit(2 * 3.141592653589793) * F.col("hour_of_day") / F.lit(24.0)))
        .withColumn("amount_x_ip_risk", F.col("amount") * F.col("ip_risk_score"))
        .withColumn("night_x_amount", F.col("is_night") * F.col("amount"))
        .withColumn("night_x_ip_risk", F.col("is_night") * F.col("ip_risk_score"))
        .withColumn("high_amount_x_high_ip", F.col("is_high_amount") * F.col("high_risk_ip_flag"))
        .withColumn("weekend_x_amount", F.col("is_weekend") * F.col("amount"))
        .withColumn("log_amount_x_ip", F.col("amount_log") * F.col("ip_risk_score"))
    )

    prior = Window.partitionBy("user_id").orderBy("event_ts").rowsBetween(Window.unboundedPreceding, -1)
    last_1h = Window.partitionBy("user_id").orderBy("event_ts").rangeBetween(-3600, -1)
    last_1d = Window.partitionBy("user_id").orderBy("event_ts").rangeBetween(-86400, -1)
    last_7d = Window.partitionBy("user_id").orderBy("event_ts").rangeBetween(-7 * 86400, -1)
    device_prior = (
        Window.partitionBy("user_id", "device_type").orderBy("event_ts").rowsBetween(Window.unboundedPreceding, -1)
    )
    category_prior = (
        Window.partitionBy("user_id", "merchant_category")
        .orderBy("event_ts")
        .rowsBetween(Window.unboundedPreceding, -1)
    )

    featured = (
        featured.withColumn("user_prior_txn_count", _count(prior))
        .withColumn("user_prior_amount_sum", _sum("amount", prior))
        .withColumn("user_prior_amount_mean", _avg("amount", prior))
        .withColumn("user_prior_amount_std", _std("amount", prior))
        .withColumn("user_prior_amount_max", _max("amount", prior))
        .withColumn("user_prior_ip_mean", _avg("ip_risk_score", prior))
        .withColumn("user_prior_ip_max", _max("ip_risk_score", prior))
        .withColumn("user_prior_night_rate", _avg("is_night", prior))
        .withColumn("user_prior_high_amount_rate", _avg("is_high_amount", prior))
        .withColumn("user_prior_high_ip_rate", _avg("high_risk_ip_flag", prior))
        .withColumn("user_prior_hour_mean", _avg("hour_of_day", prior))
        .withColumn("user_1h_txn_count", _count(last_1h))
        .withColumn("user_1h_amount_sum", _sum("amount", last_1h))
        .withColumn("user_1h_amount_mean", _avg("amount", last_1h))
        .withColumn("user_1h_ip_mean", _avg("ip_risk_score", last_1h))
        .withColumn("user_1d_txn_count", _count(last_1d))
        .withColumn("user_1d_amount_sum", _sum("amount", last_1d))
        .withColumn("user_1d_amount_mean", _avg("amount", last_1d))
        .withColumn("user_1d_amount_max", _max("amount", last_1d))
        .withColumn("user_1d_ip_mean", _avg("ip_risk_score", last_1d))
        .withColumn("user_1d_night_count", _sum("is_night", last_1d))
        .withColumn("user_1d_high_amount_count", _sum("is_high_amount", last_1d))
        .withColumn("user_1d_high_ip_count", _sum("high_risk_ip_flag", last_1d))
        .withColumn("user_7d_txn_count", _count(last_7d))
        .withColumn("user_7d_amount_sum", _sum("amount", last_7d))
        .withColumn("user_7d_amount_mean", _avg("amount", last_7d))
        .withColumn("user_7d_ip_mean", _avg("ip_risk_score", last_7d))
        .withColumn("user_device_prior_txn_count", _count(device_prior))
        .withColumn("user_device_prior_amount_mean", _avg("amount", device_prior))
        .withColumn("user_category_prior_txn_count", _count(category_prior))
        .withColumn("user_category_prior_amount_mean", _avg("amount", category_prior))
    )

    z_raw = (F.col("amount") - F.col("user_prior_amount_mean")) / F.greatest(
        F.col("user_prior_amount_std"), F.lit(1e-6)
    )
    featured = (
        featured.withColumn(
            "amount_to_user_mean_ratio",
            F.when(
                (F.col("user_prior_txn_count") > 0) & (F.col("user_prior_amount_mean") > 0),
                F.col("amount") / F.col("user_prior_amount_mean"),
            ).otherwise(F.lit(1.0)),
        )
        .withColumn(
            "amount_to_user_max_ratio",
            F.when(
                (F.col("user_prior_txn_count") > 0) & (F.col("user_prior_amount_max") > 0),
                F.col("amount") / F.col("user_prior_amount_max"),
            ).otherwise(F.lit(1.0)),
        )
        .withColumn(
            "amount_minus_user_mean",
            F.when(F.col("user_prior_txn_count") > 0, F.col("amount") - F.col("user_prior_amount_mean")).otherwise(
                F.lit(0.0)
            ),
        )
        .withColumn(
            "amount_zscore_user",
            F.when(
                (F.col("user_prior_txn_count") >= 2) & (F.col("user_prior_amount_std") > 1e-6),
                F.least(F.greatest(z_raw, F.lit(-10.0)), F.lit(10.0)),
            ).otherwise(F.lit(0.0)),
        )
        .withColumn(
            "ip_vs_user_mean",
            F.when(
                F.col("user_prior_txn_count") > 0,
                F.col("ip_risk_score") - F.col("user_prior_ip_mean"),
            ).otherwise(F.lit(0.0)),
        )
        .withColumn(
            "hour_vs_user_mean_abs",
            F.when(
                F.col("user_prior_txn_count") > 0,
                F.abs(F.col("hour_of_day") - F.col("user_prior_hour_mean")),
            ).otherwise(F.lit(0.0)),
        )
        .withColumn(
            "velocity_1h_to_1d",
            F.when(
                F.col("user_1d_txn_count") > 0,
                F.col("user_1h_txn_count") / F.col("user_1d_txn_count"),
            ).otherwise(F.lit(0.0)),
        )
        .withColumn(
            "velocity_1d_to_7d",
            F.when(
                F.col("user_7d_txn_count") > 0,
                F.col("user_1d_txn_count") / F.col("user_7d_txn_count"),
            ).otherwise(F.lit(0.0)),
        )
        .withColumn(
            "amount_1h_to_1d",
            F.when(
                F.col("user_1d_amount_sum") > 0,
                F.col("user_1h_amount_sum") / F.col("user_1d_amount_sum"),
            ).otherwise(F.lit(0.0)),
        )
        .withColumn("new_user_flag", F.when(F.col("user_prior_txn_count") == 0, F.lit(1.0)).otherwise(F.lit(0.0)))
        .withColumn(
            "device_share_of_user",
            F.when(
                F.col("user_prior_txn_count") > 0,
                F.col("user_device_prior_txn_count") / F.col("user_prior_txn_count"),
            ).otherwise(F.lit(0.0)),
        )
        .withColumn(
            "new_device_flag",
            F.when(F.col("user_device_prior_txn_count") == 0, F.lit(1.0)).otherwise(F.lit(0.0)),
        )
        .withColumn(
            "category_share_of_user",
            F.when(
                F.col("user_prior_txn_count") > 0,
                F.col("user_category_prior_txn_count") / F.col("user_prior_txn_count"),
            ).otherwise(F.lit(0.0)),
        )
        .withColumn(
            "new_category_flag",
            F.when(F.col("user_category_prior_txn_count") == 0, F.lit(1.0)).otherwise(F.lit(0.0)),
        )
    )

    featured = featured.withColumn(
        "amount_bucket",
        F.when(F.col("amount") < 50, "low")
        .when(F.col("amount") < 200, "medium")
        .when(F.col("amount") < 500, "high")
        .otherwise("very_high"),
    ).withColumn("category_device_combo", F.concat_ws("_", "merchant_category", "device_type"))

    missing = [col for col in BEHAVIORAL_COLUMNS if col not in featured.columns]
    if missing:
        raise RuntimeError(f"Spark feature set is missing {missing}")
    return featured.fillna(0, subset=BEHAVIORAL_COLUMNS)


def spark_to_pandas(df: DataFrame):
    return df.toPandas()

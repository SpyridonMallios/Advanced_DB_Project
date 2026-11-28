import sys
import time
from pyspark.sql import SparkSession
from pyspark.sql import types as T
from pyspark.sql import functions as F

# Choose the file format using the appropriate argument
if len(sys.argv) < 2 or sys.argv[1] not in ("json_schema", "json_infer", "parquet"):
    print("The command should be: spark-submit spark.py <json_schema|json_infer|parquet>")
    sys.exit(1)
MODE = sys.argv[1]


spark = (
    SparkSession.builder
    .appName(f"Benchmark_Review_Average_Score_{MODE}")
    .getOrCreate()
)
sc = spark.sparkContext


print('Spark version:', spark.version)
print('Master:', sc.master)
print('Default parallelism:', sc.defaultParallelism)
print('Shuffle partitions:', spark.conf.get('spark.sql.shuffle.partitions'))
print('AQE enabled:', spark.conf.get('spark.sql.adaptive.enabled'))


# The paths for the review data in JSON and Parquet format
JSON_YELP_REVIEW_PATH = 'hdfs:///advdb/raw/yelp_academic_dataset_review.json'
PARQUET_YELP_REVIEW_PATH = 'hdfs:///advdb/parquet/yelp_review'


# Defined schemas for the Yelp review data
review_schema = T.StructType([
    T.StructField("review_id", T.StringType(), nullable=False),
    T.StructField("user_id", T.StringType(), nullable=True),
    T.StructField("business_id", T.StringType(), nullable=True),
    T.StructField("stars", T.DoubleType(), nullable=True),
    T.StructField("useful", T.IntegerType(), nullable=True),
    T.StructField("funny", T.IntegerType(), nullable=True),
    T.StructField("cool", T.IntegerType(), nullable=True),
    T.StructField("text", T.StringType(), nullable=True),
    T.StructField("date", T.TimestampType(), nullable=True)
])


# Create the dataframe based on which format is given as argument
if MODE == "json_schema":
    df_review = (
        spark.read
        .schema(review_schema)
        .option("timestampFormat", "yyyy-MM-dd HH:mm:ss")
        .json(JSON_YELP_REVIEW_PATH)
        .withColumn("year", F.year(F.col("date")))
    )
elif MODE == "json_infer":
    # Normally reading is a lazy transformation, so the time counting should not be started yet.
    # However, with inferSchema = True, a Spark job begins immediately, so the time counting must be started. 
    t0 = time.perf_counter()
    df_review = (
        spark.read
        .option("inferSchema", "true")
        .option("timestampFormat", "yyyy-MM-dd HH:mm:ss")
        .json(JSON_YELP_REVIEW_PATH)
        .withColumn("year", F.year(F.col("date")))
    )
elif MODE == "parquet":
    df_review = spark.read.parquet(PARQUET_YELP_REVIEW_PATH)


# Create the query for finding average review score per year
df_query = (
    df_review.groupBy("year")
    .agg(F.avg("stars").alias("avg_stars"))
    .orderBy("year")
)


if MODE != "json_infer":
    t0 = time.perf_counter()

# Count the time it takes for the Spark job to finish the collect() function
results = df_query.collect()
t1 = time.perf_counter()
elapsed = t1 - t0


# Print the query results
print(f"\n=== Query results ({MODE}) ===")
for row in results:
    print(f"Year: {row['year']} | Avg Stars: {row['avg_stars']:.4f}")


# Print the explanation of the Physical Plan and the execution time
print(f"\n=== Explain ({MODE}) ===")
df_query.explain(mode="formatted")
print(f"\n>>> [{MODE}] Execution time: {elapsed:.4f} seconds <<<\n")


spark.stop()
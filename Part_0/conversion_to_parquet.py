from pyspark.sql import SparkSession
from pyspark.sql import types as T
from pyspark.sql import functions as F


spark = (
    SparkSession.builder
    .appName("Spark Parquet Conversion")
    .getOrCreate()
)


sc = spark.sparkContext


print('Spark version:', spark.version)
print('Master:', sc.master)
print('Default parallelism:', sc.defaultParallelism)
print('Shuffle partitions:', spark.conf.get('spark.sql.shuffle.partitions'))
print('AQE enabled:', spark.conf.get('spark.sql.adaptive.enabled'))


# The paths for the data in HDFS
YELP_BUSINESS_PATH = 'hdfs:///advdb/raw/yelp_academic_dataset_business.json'
YELP_REVIEW_PATH = 'hdfs:///advdb/raw/yelp_academic_dataset_review.json'
YELP_USER_PATH = 'hdfs:///advdb/raw/yelp_academic_dataset_user.json'
GLOVE_PATH = 'hdfs:///advdb/raw/glove.6B.100d.txt'
ACS_PATH = 'hdfs:///advdb/raw/acs_zcta.csv'


# Output paths for Parquet files in HDFS
OUT_YELP_BUSINESS_PATH = 'hdfs:///advdb/parquet/yelp_business'
OUT_YELP_REVIEW_PATH = 'hdfs:///advdb/parquet/yelp_review'
OUT_YELP_USER_PATH = 'hdfs:///advdb/parquet/yelp_user'
OUT_GLOVE_PATH = 'hdfs:///advdb/parquet/glove_100d'
OUT_ACS_PATH = 'hdfs:///advdb/parquet/acs_zcta'


# Defined schemas for the Yelp data
business_schema = T.StructType([
    T.StructField("business_id", T.StringType(), nullable=False),
    T.StructField("name", T.StringType(), nullable=True),
    T.StructField("city", T.StringType(), nullable=True),
    T.StructField("state", T.StringType(), nullable=True),
    T.StructField("postal_code", T.StringType(), nullable=True),
    T.StructField("latitude", T.DoubleType(), nullable=True),
    T.StructField("longitude", T.DoubleType(), nullable=True),
    T.StructField("stars", T.DoubleType(), nullable=True),
    T.StructField("review_count", T.IntegerType(), nullable=True),
    T.StructField("is_open", T.IntegerType(), nullable=True),
    T.StructField("attributes", T.MapType(T.StringType(), T.StringType()), nullable=True),
    T.StructField("hours", T.MapType(T.StringType(), T.StringType()), nullable=True)
])


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


user_schema = T.StructType([
    T.StructField("user_id", T.StringType(), nullable=False),
    T.StructField("review_count", T.IntegerType(), nullable=True),
    T.StructField("yelping_since", T.TimestampType(), nullable=True),
    T.StructField("useful", T.IntegerType(), nullable=True),
    T.StructField("funny", T.IntegerType(), nullable=True),
    T.StructField("cool", T.IntegerType(), nullable=True),
    T.StructField("elite", T.StringType(), nullable=True),
    T.StructField("friends", T.StringType(), nullable=True),
    T.StructField("fans", T.IntegerType(), nullable=True),
    T.StructField("average_stars", T.DoubleType(), nullable=True),
    T.StructField("compliment_hot", T.IntegerType(), nullable=True),
    T.StructField("compliment_more", T.IntegerType(), nullable=True),
    T.StructField("compliment_profile", T.IntegerType(), nullable=True),
    T.StructField("compliment_cute", T.IntegerType(), nullable=True),
    T.StructField("compliment_list", T.IntegerType(), nullable=True),
    T.StructField("compliment_note", T.IntegerType(), nullable=True),
    T.StructField("compliment_plain", T.IntegerType(), nullable=True),
    T.StructField("compliment_cool", T.IntegerType(), nullable=True),
    T.StructField("compliment_funny", T.IntegerType(), nullable=True),
    T.StructField("compliment_writer", T.IntegerType(), nullable=True),
    T.StructField("compliment_photos", T.IntegerType(), nullable=True)
])


# Change the JSON files to Parquet
df_business = spark.read.schema(business_schema).json(YELP_BUSINESS_PATH)
df_business.write.mode("overwrite").parquet(OUT_YELP_BUSINESS_PATH)


df_review = spark.read \
    .schema(review_schema) \
    .option("timestampFormat", "yyyy-MM-dd HH:mm:ss") \
    .json(YELP_REVIEW_PATH)
df_review_partitioned = df_review.withColumn("year", F.year(F.col("date")))
df_review_partitioned.write \
    .mode("overwrite") \
    .partitionBy("year") \
    .parquet(OUT_YELP_REVIEW_PATH)


df_user = spark.read \
    .schema(user_schema) \
    .option("timestampFormat", "yyyy-MM-dd HH:mm:ss") \
    .json(YELP_USER_PATH)
df_user.write.mode("overwrite").parquet(OUT_YELP_USER_PATH)


# The first part is the word and the next 100 numbers is the dimension vector
df_glove_raw = spark.read.text(GLOVE_PATH)
split_col = F.split(F.trim(F.col("value")), r"\s+")

df_glove = df_glove_raw.select(
    split_col.getItem(0).cast(T.StringType()).alias("word"),
    F.slice(split_col, 2, 100).cast(T.ArrayType(T.FloatType())).alias("vec")
)
df_glove.write.mode("overwrite").parquet(OUT_GLOVE_PATH)


# Replace zcta field type to be a String type
acs_base_schema = spark.read \
    .option("header", "true") \
    .option("inferSchema", "true") \
    .csv(ACS_PATH) \
    .schema

acs_explicit_schema = T.StructType([
    T.StructField("zcta", T.StringType(), nullable=True)
    if field.name == "zcta"
    else field
    for field in acs_base_schema
])

df_acs = spark.read \
    .option("header", "true") \
    .schema(acs_explicit_schema) \
    .csv(ACS_PATH)
df_acs.write.mode("overwrite").parquet(OUT_ACS_PATH)

print("Η μετατροπή των δεδομένων σε Parquet είναι επιτυχής!")
spark.stop()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug  4 19:22:30 2024

@author: musthafa
"""

import os
import findspark
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk

# Download NLTK data
nltk.download('vader_lexicon')

# Set up environment and find Spark
os.environ['PYSPARK_SUBMIT_ARGS'] = '--packages org.apache.spark:spark-streaming-kafka-0-10_2.12:3.2.0,org.apache.spark:spark-sql-kafka-0-10_2.12:3.2.0 pyspark-shell'
findspark.init()

# Initialize Spark session
spark = SparkSession.builder \
    .appName("drilling-anomalie-detection") \
    .master("local[*]") \
    .getOrCreate()

# Initialize sentiment analysis model
sia = SentimentIntensityAnalyzer()

# Define schema for incoming data
json_schema = StructType([
    StructField("TIME", StringType(), True),
    StructField("SPPA", StringType(), True),
    StructField("ROP30s", StringType(), True),
    StructField("TQ30s", StringType(), True),
    StructField("ECD_MW_IN", StringType(), True),
])

# Create a streaming DataFrame from Kafka
df = spark \
    .readStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", "localhost:9092") \
    .option("subscribe", "drilling-well-0001") \
    .option("startingOffsets", "earliest") \
    .load()

# Parse the JSON data and extract the features
json_df = df.select(from_json(col("value").cast("string"), json_schema).alias("value"))

# Extract text for sentiment analysis (adjust this based on your data)
text_df = json_df.select(col("value.SPPA").alias("text"))

# Define UDF for sentiment analysis
def get_sentiment(text):
    print(text)
    if text is None:
        return None
    sentiment_score = sia.polarity_scores(text)
    return sentiment_score['compound']

sentiment_udf = udf(get_sentiment, FloatType())

# Apply the UDF to get sentiment scores
sentiment_df = text_df.withColumn("sentiment_score", sentiment_udf(col("text")))

# Output the sentiment analysis results to console
query = sentiment_df.writeStream \
    .format("console") \
    .outputMode("append") \
    .start()

print(query)

query.awaitTermination()

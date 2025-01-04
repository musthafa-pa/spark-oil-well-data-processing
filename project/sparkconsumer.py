#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Updated for unified SQL structure with batch and prediction support.
"""

import os
import findspark
import pandas as pd
import psycopg2
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *
import joblib
from random_forest_algorithm_with_sample_data import DrillingFaultPredictor

# Load the trained model
predictor = joblib.load('optimized_model.joblib')

# Set up Spark and Kafka dependencies
os.environ['PYSPARK_SUBMIT_ARGS'] = '--packages org.apache.spark:spark-streaming-kafka-0-10_2.12:3.2.0,org.apache.spark:spark-sql-kafka-0-10_2.12:3.2.0 pyspark-shell'
findspark.init()

# Initialize Spark session
spark = SparkSession.builder \
    .appName("drilling-anomaly-detection") \
    .master("local[*]") \
    .getOrCreate()

# Define schema for incoming data
json_schema = StructType([
    StructField("TIME", TimestampType(), True),
    StructField("SPPA", StringType(), True),
    StructField("CPPA", StringType(), True),
    StructField("ROP", StringType(), True),
])

# Subscribe to multiple topics
topics = "drilling-well-0001,drilling-well-0002,drilling-well-0003,drilling-well-0004,drilling-well-0005"

df = spark \
    .readStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", "localhost:9092") \
    .option("subscribe", topics) \
    .option("startingOffsets", "latest") \
    .load()

# Parse JSON data and include the `topic` field for differentiation
json_df = df.select(
    col("topic").alias("topic"),
    from_json(col("value").cast("string"), json_schema).alias("value")
)

features_df = json_df.select(
    col("topic"),
    col("value.TIME").alias("TIME"),
    col("value.SPPA").alias("SPPA"),
    col("value.CPPA").alias("CPPA"),
    col("value.ROP").alias("ROP")
).na.fill({
    "SPPA": 0.0,  # Default value for missing SPPA
    "CPPA": 0.0,  # Default value for missing CPPA
    "ROP": 0.0    # Default value for missing ROP
})
    
# Database connection setup
def get_db_connection():
    conn = psycopg2.connect(
        dbname="postgres", user="musthafa", password="", host="localhost", port="5432"
    )
    return conn

# Insert data into Well_Data table
def insert_batch_data(conn, well_id, batch_no, batch_data, prediction):
    cursor = conn.cursor()
    insert_query = """
    INSERT INTO Well_Data (well_id, batch_no, sppa, cppa, rop, time, prediction)
    VALUES (%s, %s, %s, %s, %s, %s, %s)
    """
    for row in batch_data:
        # Convert numpy.int64 or numpy.float64 to native Python types
        well_id = int(well_id)
        batch_no = int(batch_no)
        sppa = float(row['SPPA'])  # Ensure sppa is a float
        cppa = float(row['CPPA'])  # Ensure cppa is a float
        rop = float(row['ROP'])    # Ensure rop is a float
        print(f"sppa: {sppa} CPPA: {cppa} ROP: {rop}")
        time = row['TIME']         # Assuming TIME is already a valid datetime or timestamp type
        prediction = str(prediction)  # Ensure prediction is a string

        # Insert the data into the table
        cursor.execute(insert_query, (well_id, batch_no, sppa, cppa, rop, time, prediction))

    conn.commit()

# Process rows from each topic
accumulators = {}
def process_row(topic, row):
    # Extract well_id from the topic name
    well_id = int(topic.split('-')[-1])

    # Ensure an accumulator exists for the topic
    if topic not in accumulators:
        accumulators[topic] = {'data': [], 'batch_no': 1}

    accumulator = accumulators[topic]

    # Add the current row to the accumulator
    accumulator['data'].append({
        'TIME': row['TIME'],
        'SPPA': row['SPPA'],
        'CPPA': row['CPPA'],
        'ROP': row['ROP']
    })

    # Process if the accumulator contains 10 rows
    if len(accumulator['data']) == 10:
        # Extract batch data
        batch_data = accumulator['data']
        
        #print(f"batch_data: {batch_data}")

        # Run prediction
        features = pd.DataFrame(batch_data)
        prediction = predictor.predict_majority_for_entire_data(features[['SPPA', 'CPPA', 'ROP']].to_dict(orient='list'))

        # Insert batch data into the database
        conn = get_db_connection()
        try:
            insert_batch_data(conn, well_id, accumulator['batch_no'], batch_data, prediction)
        finally:
            conn.close()

        # Reset the accumulator for the next batch
        accumulator['data'] = []
        accumulator['batch_no'] += 1

# Process batches
def process_batch(df, batch_id):
    df.select("SPPA").show(5, False)

    batch_df = df.toPandas()
    print(f"batch_df{batch_df}")
    for row in batch_df.itertuples(index=False):
        topic = row.topic
        #print(f"topic::{row}")
        process_row(topic, row._asdict())

# Start the streaming query
query = features_df.writeStream \
    .outputMode("append") \
    .foreachBatch(process_batch) \
    .start()

query.awaitTermination()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug  4 19:22:30 2024
"""

import os
import findspark
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *
import pickle
from random_forest_algorithm_with_sample_data import DrillingFaultPredictor  # Import your model class

# Set up environment and find Spark
os.environ['PYSPARK_SUBMIT_ARGS'] = '--packages org.apache.spark:spark-streaming-kafka-0-10_2.12:3.2.0,org.apache.spark:spark-sql-kafka-0-10_2.12:3.2.0 pyspark-shell'
findspark.init()

# Initialize Spark session
spark = SparkSession.builder \
    .appName("drilling-anomalie-detection") \
    .master("local[*]") \
    .getOrCreate()

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
features_df = json_df.select(
    col("value.SPPA").cast("double").alias("SPPA"),
    col("value.ROP30s").cast("double").alias("ROP30s"),
    col("value.TQ30s").cast("double").alias("TQ30s"),
    col("value.ECD_MW_IN").cast("double").alias("ECD_MW_IN")
)

# Load the trained model (assumes model is saved as 'model.pkl')
with open('model.pkl', 'rb') as file:
    predictor = pickle.load(file)

def predict(features):
    # Convert features to DataFrame
    features_df = pd.DataFrame(features)
    # Predict using the model
    predictions = predictor.predict_new_data(features_df)
    return predictions

# Initialize an accumulator to store the features
accumulator = {
    'SPPA': [],
    'ROP30s': [],
    'TQ30s': [],
    'ECD_MW_IN': []
}

def process_row(row):
    # Add the new row's features to the accumulator
    accumulator['SPPA'].append(row['SPPA'])
    accumulator['ROP30s'].append(row['ROP30s'])
    accumulator['TQ30s'].append(row['TQ30s'])
    accumulator['ECD_MW_IN'].append(row['ECD_MW_IN'])
    
    # Check if we have 10 data points accumulated
    if len(accumulator['SPPA']) >= 10:
        print("Processing accumulated data...")
        # Get predictions for the accumulated data
        features = {
            'SPPA': accumulator['SPPA'],
            'ROP30s': accumulator['ROP30s'],
            'TQ30s': accumulator['TQ30s'],
            'ECD_MW_IN': accumulator['ECD_MW_IN']
        }
        print(f'Raw features from accumulated data: {features}')
        
        # Get the prediction from the model
        prediction = predict(features)
        print(f'Prediction for accumulated data: {prediction}')
        
        # Reset the accumulator after processing
        reset_data()

def reset_data():
    global accumulator
    accumulator = {
        'SPPA': [],
        'ROP30s': [],
        'TQ30s': [],
        'ECD_MW_IN': []
    }

# Collect streaming data and process it
def process_batch(df, batch_id):
    # Convert the Spark DataFrame batch to a Pandas DataFrame for further processing
    batch_df = df.toPandas()
    print(f'Processing batch: {batch_id}')
    
    # Iterate through each row of the batch and run the prediction
    for row in batch_df.itertuples(index=False):
        process_row(row._asdict())

# Start the streaming query, processing each batch as it comes in
query = features_df.writeStream \
    .outputMode("append") \
    .foreachBatch(process_batch) \
    .start()

query.awaitTermination()

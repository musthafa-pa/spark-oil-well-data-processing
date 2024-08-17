#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug  4 19:22:30 2024

@author: musthafa
"""


import os
from pyspark.sql import SparkSession
import findspark

os.environ['PYSPARK_SUBMIT_ARGS'] = '--packages org.apache.spark:spark-streaming-kafka-0-10_2.12:3.2.0,org.apache.spark:spark-sql-kafka-0-10_2.12:3.2.0 pyspark-shell'
findspark.init()



spark = SparkSession.builder \
    .appName("drilling-anomalie-detection") \
    .master("local[*]") \
    .getOrCreate()


df = spark \
  .readStream \
  .format("kafka") \
  .option("kafka.bootstrap.servers", "localhost:9092") \
  .option("subscribe", "drillinglog") \
  .option("startingOffsets", "earliest") \
  .load()


query = df \
    .selectExpr("CAST(key AS STRING)", "CAST(value AS STRING)") \
    .writeStream \
    .format("console") \
    .outputMode("append") \
    .start()

query.awaitTermination()

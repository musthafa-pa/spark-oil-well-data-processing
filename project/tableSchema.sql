#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  3 14:18:17 2025

@author: musthafa
"""

CREATE TABLE Well_Logs (
    well_id INT NOT NULL,
    CPPA NUMERIC(10, 2),
    SPPA NUMERIC(10, 2),
    ROP NUMERIC(10, 2),
    TIME TIMESTAMP NOT NULL,
    Batch_No INT NOT NULL,
    PRIMARY KEY (well_id, TIME),
    CONSTRAINT unique_batch_no UNIQUE (Batch_No)  -- Ensure Batch_No is unique
);

CREATE TABLE Predictions (
    well_id INT NOT NULL,
    Batch_No INT NOT NULL,
    Prediction VARCHAR(255),
    PRIMARY KEY (well_id, Batch_No),
    FOREIGN KEY (Batch_No) REFERENCES Well_Logs (Batch_No)
        ON DELETE CASCADE
);
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 17 13:40:49 2024

@author: musthafa
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.inspection import partial_dependence
import matplotlib.pyplot as plt

# Sample dataset
data = {
    'SPP': [3000, 3100, 2900, 3200, 3150, 3300, 3000, 2950, 3400, 3250],
    'ROP': [50, 55, 53, 47, 45, 60, 58, 52, 49, 50],
    'Torque': [1000, 1050, 980, 1100, 1080, 1150, 1020, 995, 1180, 1110],
    'Mud_Weight': [10, 10.5, 9.8, 11, 10.8, 11.5, 10.2, 9.9, 11.8, 11.1],
    'Failure': [0, 0, 0, 1, 0, 1, 0, 0, 1, 1]
}

df = pd.DataFrame(data)

# Features and labels
X = df[['SPP', 'ROP', 'Torque', 'Mud_Weight']]
y = df['Failure']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize the model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
rf_model.fit(X_train, y_train)

# Predictions on the test set
y_pred = rf_model.predict(X_test)

# Evaluate the model
print(f'Accuracy: {accuracy_score(y_test, y_pred)}')
print(f'Classification Report:\n{classification_report(y_test, y_pred)}')

# Feature importance
importances = rf_model.feature_importances_
feature_names = X.columns
importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
print(f'Feature Importances:\n{importance_df}')

# Plot feature importance
plt.figure(figsize=(10, 6))
importance_df.sort_values('Importance', ascending=False).plot(kind='bar', x='Feature', y='Importance', legend=False)
plt.title('Feature Importances')
plt.xlabel('Feature')
plt.ylabel('Importance')
plt.show()

# Partial Dependence Plots
fig, ax = plt.subplots(figsize=(12, 8))
features = ['SPP', 'ROP', 'Torque', 'Mud_Weight']
for feature in features:
    pdp, values = partial_dependence(rf_model, X_train, [feature], kind='both')
    ax.plot(values[0], pdp[0], label=feature)

ax.set_title('Partial Dependence Plots')
ax.set_xlabel('Feature Value')
ax.set_ylabel('Partial Dependence')
ax.legend()
plt.show()

# New data for prediction
new_data = {
    'SPP': [3100, 3200, 3150],
    'ROP': [52, 48, 50],
    'Torque': [1050, 1120, 1085],
    'Mud_Weight': [10.4, 11.2, 10.9]
}
new_df = pd.DataFrame(new_data)

# Predict
new_predictions = rf_model.predict(new_df)
print(f'Predictions: {new_predictions}')

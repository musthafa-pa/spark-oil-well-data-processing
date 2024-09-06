#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 17 13:40:49 2024

@author: musthafa
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.inspection import PartialDependenceDisplay
import matplotlib.pyplot as plt
import pickle

class DrillingFaultPredictor:
    def __init__(self, data):
        self.df = pd.DataFrame(data)
        self.features = ['SPPA', 'ROP30s', 'TQ30s', 'ECD_MW_IN']
        self.target = 'Failure'
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
    
    def preprocess_data(self):
        # Split data into features and target
        X = self.df[self.features]
        y = self.df[self.target]
        # Split data into training and testing sets
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    def train_model(self):
        # Train the model
        self.model.fit(self.X_train, self.y_train)
    
    def evaluate_model(self):
        # Predictions on the test set
        y_pred = self.model.predict(self.X_test)
        # Evaluate the model
        print(f'Accuracy: {accuracy_score(self.y_test, y_pred)}')
        print(f'Classification Report:\n{classification_report(self.y_test, y_pred)}')
    
    def plot_feature_importance(self):
        # Feature importance
        importances = self.model.feature_importances_
        importance_df = pd.DataFrame({'Feature': self.features, 'Importance': importances})
        print(f'Feature Importances:\n{importance_df}')
        
        # Plot feature importance
        plt.figure(figsize=(10, 6))
        importance_df.sort_values('Importance', ascending=False).plot(kind='bar', x='Feature', y='Importance', legend=False)
        plt.title('Feature Importances')
        plt.xlabel('Feature')
        plt.ylabel('Importance')
        plt.show()
    
    def plot_partial_dependence(self):
        # Partial Dependence Plots
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Plot the first feature to initialize the display
        display = PartialDependenceDisplay.from_estimator(self.model, self.X_train, [self.features[0]], ax=ax)
        
        # Loop over the remaining features and add them to the display
        for feature in self.features[1:]:
            display = PartialDependenceDisplay.from_estimator(self.model, self.X_train, [feature], ax=display.axes_)
        
        ax.set_title('Partial Dependence Plots')
        ax.set_xlabel('Feature Value')
        ax.set_ylabel('Partial Dependence')
        ax.legend(self.features)
        plt.show()
    
    def save_model(self, filename):
        # Save the entire predictor instance to a file
        with open(filename, 'wb') as file:
            pickle.dump(self, file)
        print(f"Model saved to {filename}")
    
    def predict_new_data(self, new_data):
        print(f'newdata {new_data}')
        # Convert new data to DataFrame
        new_df = pd.DataFrame(new_data)
        
        # Ensure the new data has the same features
        if set(self.features).difference(new_df.columns):
            raise ValueError("New data does not contain all required features.")
        
        # Predict using the model
        predictions = self.model.predict(new_df[self.features])
        return predictions

# Example usage:

# Original labelled training data
data = {
    'SPPA': [3000, 3100, 2900, 3200, 3150, 3300, 3000, 2950, 3400, 3250],
    'ROP30s': [50, 55, 53, 47, 45, 60, 58, 52, 49, 50],
    'TQ30s': [1000, 1050, 980, 1100, 1080, 1150, 1020, 995, 1180, 1110],
    'ECD_MW_IN': [10, 10.5, 9.8, 11, 10.8, 11.5, 10.2, 9.9, 11.8, 11.1],
    'Failure': [0, 0, 0, 1, 0, 1, 0, 0, 1, 1]
}

# Instantiate and use the model
predictor = DrillingFaultPredictor(data)

# Preprocess the data
predictor.preprocess_data()

# Train the model
predictor.train_model()

# Evaluate the model
predictor.evaluate_model()

# Plot feature importance
predictor.plot_feature_importance()

# Plot partial dependence
predictor.plot_partial_dependence()

# Save the model
predictor.save_model('model.pkl')

# New data for prediction (can be used later)
new_data = {
    'SPPA': [3100, 3200, 3150],
    'ROP30s': [52, 48, 50],
    'TQ30s': [1050, 1120, 1085],
    'ECD_MW_IN': [10.4, 11.2, 10.9]
}
predictions = predictor.predict_new_data(new_data)
print(f'Predictions: {predictions}')

import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import joblib
import logging
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DrillingFaultPredictor:
    def __init__(self, data):
        self.df = pd.DataFrame(data)
        self.features = ['SPPA', 'CPPA', 'ROP']
        self.target = 'Failure'
        self.model = RandomForestClassifier(random_state=42)

    def preprocess_data(self):
        X = self.df[self.features]
        y = self.df[self.target]

        # Split data into training and testing sets
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )
        logging.info("Data preprocessing complete. Training and testing sets created.")

    def tune_hyperparameters(self):
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }

        grid_search = GridSearchCV(
            estimator=self.model, param_grid=param_grid, cv=5, scoring='accuracy', verbose=2, n_jobs=-1
        )
        grid_search.fit(self.X_train, self.y_train)

        # Update model with best parameters
        self.model = grid_search.best_estimator_
        logging.info(f"Best parameters: {grid_search.best_params_}")

    def train_model(self):
        self.model.fit(self.X_train, self.y_train)
        logging.info("Model training complete.")

    def evaluate_model(self):
        y_pred = self.model.predict(self.X_test)
        y_proba = self.model.predict_proba(self.X_test)[:, 1]

        # Metrics
        accuracy = accuracy_score(self.y_test, y_pred)
        auc_score = roc_auc_score(self.y_test, y_proba)

        logging.info(f"Accuracy: {accuracy * 100:.2f}%")
        logging.info(f"ROC-AUC Score: {auc_score:.2f}")
        logging.info(f"Classification Report:\n{classification_report(self.y_test, y_pred)}")

        # Plot ROC Curve
        fpr, tpr, _ = roc_curve(self.y_test, y_proba)
        plt.figure()
        plt.plot(fpr, tpr, label=f"ROC curve (area = {auc_score:.2f})")
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend(loc="lower right")
        plt.show()

    def plot_feature_importance(self):
        importances = self.model.feature_importances_
        importance_df = pd.DataFrame({'Feature': self.features, 'Importance': importances})
        logging.info(f"Feature Importances:\n{importance_df}")

        # Plot feature importance
        plt.figure(figsize=(10, 6))
        importance_df.sort_values('Importance', ascending=False).plot(kind='bar', x='Feature', y='Importance', legend=False)
        plt.title('Feature Importances')
        plt.xlabel('Feature')
        plt.ylabel('Importance')
        plt.show()

    def save_model(self, filename):
        joblib.dump(self, filename)
        logging.info(f"Model saved to {filename}")

    def predict_majority_for_entire_data(self, new_data):
        new_df = pd.DataFrame(new_data)
        
        # Ensure the new data contains all required features
        missing_features = set(self.features) - set(new_df.columns)
        if missing_features:
            raise ValueError(f"New data is missing features: {missing_features}")
    

        # Ensure the new data has the same features
        if set(self.features).difference(new_df.columns):
            raise ValueError("New data does not contain all required features.")

        # Make predictions for each row in the new data
        predictions = self.model.predict(new_df[self.features])

        # Use majority vote to return a single prediction (0 or 1)
        majority_prediction = np.bincount(predictions).argmax()  # 0 or 1 based on majority
        return majority_prediction

# Example usage
if __name__ == "__main__":
    data = {
        'SPPA': [180, 200, 270, 200, 180, 200, 220, 500, 600, 700],
        'CPPA': [50, 55, 53, 47, 45, 60, 58, 52, 49, 50],
        'ROP': [53.1, 53.1, 53.1, 53.1, 53.1, 53.1,53.1, 53.1, 53.1, 53.1],
        'Failure': [0, 0, 0, 1, 0, 1, 0, 0, 1, 1]
    }

    predictor = DrillingFaultPredictor(data)

    # Preprocess the data
    predictor.preprocess_data()

    # Tune hyperparameters
    predictor.tune_hyperparameters()

    # Train the model
    predictor.train_model()

    # Evaluate the model
    predictor.evaluate_model()

    # Plot feature importance
    predictor.plot_feature_importance()

    # Save the model
    predictor.save_model('optimized_model.joblib')

    # Predict majority class for new data (entire dataset)
    new_data = {
        'SPPA': [300, 320, 240, 360, 380, 400],
        'CPPA': [52, 48, 50, 48, 50, 48],
        'ROP': [53.1, 53.1, 53.1, 53.1, 53.1, 53.1],
        'Failure': [None, None, None, None, None, None]  # Placeholder for target variable
    }

    majority_prediction = predictor.predict_majority_for_entire_data(new_data)
    logging.info(f"Majority Prediction for Entire Data: {majority_prediction}")  # It will print either 0 or 1

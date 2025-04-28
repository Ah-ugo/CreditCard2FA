import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import joblib
from pathlib import Path
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class FraudDetectionModel:
    """A wrapper class for the fraud detection model with proper initialization and prediction"""

    def __init__(self, model_path="fraud_detection_model.joblib"):
        self.model_path = model_path
        self.model = None
        self.scaler = None
        self.is_trained = False
        self.initialize_model()

    def initialize_model(self):
        """Initialize or load the fraud detection model"""
        try:
            if Path(self.model_path).exists():
                logger.info(f"Loading existing model from {self.model_path}")
                model_data = joblib.load(self.model_path)
                self.model = model_data.get('model')
                self.scaler = model_data.get('scaler')
                self.is_trained = model_data.get('is_trained', False)
                logger.info(f"Model loaded successfully. Trained: {self.is_trained}")
            else:
                logger.info("No existing model found. Creating a new model.")
                self.model = IsolationForest(
                    n_estimators=100,
                    contamination=0.1,  # Expected proportion of anomalies
                    random_state=42,
                    n_jobs=-1  # Use all available cores
                )
                self.scaler = StandardScaler()
                self.is_trained = False
                self.save_model()
        except Exception as e:
            logger.error(f"Error initializing model: {str(e)}")
            # Fallback to a new model
            self.model = IsolationForest(n_estimators=100, contamination=0.1, random_state=42)
            self.scaler = StandardScaler()
            self.is_trained = False

    def preprocess_features(self, features):
        """Preprocess features for model input"""
        # Convert dict to DataFrame for easier handling
        if isinstance(features, dict):
            features = pd.DataFrame([features])

        # Select only numeric columns for scaling
        numeric_features = features.select_dtypes(include=[np.number])

        if self.is_trained:
            # Use the fitted scaler
            scaled_features = self.scaler.transform(numeric_features)
        else:
            # Just return the original features if not trained
            scaled_features = numeric_features.values

        return scaled_features

    def fit(self, features):
        """Train the model with transaction data"""
        try:
            if features.shape[0] < 10:
                logger.warning("Not enough samples to train the model properly")
                return False

            logger.info(f"Training model with {features.shape[0]} samples")

            # Fit the scaler
            self.scaler.fit(features)

            # Scale the features
            scaled_features = self.scaler.transform(features)

            # Fit the model
            self.model.fit(scaled_features)
            self.is_trained = True

            # Save the trained model
            self.save_model()
            logger.info("Model trained and saved successfully")
            return True
        except Exception as e:
            logger.error(f"Error training model: {str(e)}")
            return False

    def predict(self, features):
        """Predict if a transaction is fraudulent"""
        try:
            # If model is not trained, return a default prediction
            if not self.is_trained:
                logger.warning("Model not trained yet. Returning default prediction.")
                return 1, 0.5  # Default: normal transaction, neutral score

            # Preprocess features
            processed_features = self.preprocess_features(features)

            # Get prediction (-1 for anomaly, 1 for normal)
            prediction = self.model.predict(processed_features)[0]

            # Get anomaly score
            score = self.model.decision_function(processed_features)[0]

            # Normalize score to be between 0 and 1 (higher = more normal)
            # For Isolation Forest, decision_function returns higher values for normal points
            normalized_score = (score + 0.5) / 1.0  # Adjust based on your model's output range

            return prediction, normalized_score
        except Exception as e:
            logger.error(f"Error making prediction: {str(e)}")
            return 1, 0.5  # Default in case of error

    def save_model(self):
        """Save the model to disk"""
        try:
            model_data = {
                'model': self.model,
                'scaler': self.scaler,
                'is_trained': self.is_trained
            }
            joblib.dump(model_data, self.model_path)
            logger.info(f"Model saved to {self.model_path}")
            return True
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            return False


# Function to create a sample dataset for initial training
def create_sample_dataset(n_samples=1000):
    """Create a sample dataset for initial model training"""
    # Generate normal transactions
    normal_samples = int(n_samples * 0.9)  # 90% normal
    fraud_samples = n_samples - normal_samples  # 10% fraudulent

    # Features for normal transactions
    normal_amounts = np.random.uniform(10, 1000, normal_samples)
    normal_hour = np.random.randint(8, 22, normal_samples)  # Business hours
    normal_frequency = np.random.uniform(0, 3, normal_samples)  # Low frequency

    # Features for fraudulent transactions
    fraud_amounts = np.random.uniform(500, 5000, fraud_samples)  # Higher amounts
    fraud_hour = np.random.randint(0, 24, fraud_samples)  # Any hour
    fraud_frequency = np.random.uniform(3, 10, fraud_samples)  # Higher frequency

    # Combine features
    X = np.vstack([
        np.column_stack((normal_amounts, normal_hour, normal_frequency)),
        np.column_stack((fraud_amounts, fraud_hour, fraud_frequency))
    ])

    # Create labels (not used by Isolation Forest but useful for evaluation)
    y = np.hstack([np.zeros(normal_samples), np.ones(fraud_samples)])

    return X, y


# Test the model
if __name__ == "__main__":
    # Create a sample dataset
    X, y = create_sample_dataset()

    # Initialize the model
    model = FraudDetectionModel()

    # Train the model
    model.fit(X)

    # Test prediction
    test_features = {
        'amount': 2000,
        'hour_of_day': 3,
        'frequency': 5
    }

    prediction, score = model.predict(test_features)
    print(f"Prediction: {'Fraudulent' if prediction == -1 else 'Normal'}")
    print(f"Score: {score}")

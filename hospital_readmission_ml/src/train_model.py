# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import joblib
import os

class ModelTrainer:
    def __init__(self):
        self.model = None
        self.metrics = {}
    
    def train(self, X, y, test_size=0.2, random_state=42):
        """Train the model"""
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state
            )
            
            self.model = RandomForestClassifier(n_estimators=100, random_state=random_state, n_jobs=-1)
            self.model.fit(X_train, y_train)
            
            y_pred = self.model.predict(X_test)
            y_pred_proba = self.model.predict_proba(X_test)[:, 1]
            
            self.metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred),
                'recall': recall_score(y_test, y_pred),
                'f1': f1_score(y_test, y_pred),
                'auc_roc': roc_auc_score(y_test, y_pred_proba)
            }
            
            print(f"✓ Model trained successfully")
            print(f"  - Accuracy: {self.metrics['accuracy']:.4f}")
            print(f"  - Precision: {self.metrics['precision']:.4f}")
            print(f"  - Recall: {self.metrics['recall']:.4f}")
            print(f"  - F1-Score: {self.metrics['f1']:.4f}")
            print(f"  - AUC-ROC: {self.metrics['auc_roc']:.4f}")
            
            return self.model, self.metrics
        except Exception as e:
            print(f"✗ Training error: {e}")
            raise
    
    def save_model(self, model_path):
        """Save trained model"""
        try:
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            joblib.dump(self.model, model_path)
            print(f"✓ Model saved at {model_path}")
        except Exception as e:
            print(f"✗ Error saving model: {e}")
            raise

# Load the dataset
def load_data(file_path):
    data = pd.read_csv(file_path)
    return data

# Preprocess the data
def preprocess_data(data):
    # Example preprocessing steps
    # Convert categorical variables to dummy variables
    data = pd.get_dummies(data, drop_first=True)
    return data

if __name__ == "__main__":
    # Define file paths
    raw_data_path = '../data/raw/hospital_readmissions.csv'
    model_path = '../models/xgb_readmission.pkl'
    
    # Load and preprocess data
    data = load_data(raw_data_path)
    processed_data = preprocess_data(data)
    
    # Separate features and target
    X = processed_data.drop('readmission', axis=1)  # Assuming 'readmission' is the target variable
    y = processed_data['readmission']
    
    # Initialize and train the model
    trainer = ModelTrainer()
    model, metrics = trainer.train(X, y)
    
    # Save the trained model
    trainer.save_model(model_path)
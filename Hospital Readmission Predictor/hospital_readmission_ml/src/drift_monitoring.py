# Drift Monitoring Module

import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import joblib
import os

class DriftMonitor:
    def __init__(self, model_path, reference_data_path):
        try:
            model_path = os.path.abspath(model_path)
            reference_data_path = os.path.abspath(reference_data_path)
            
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found: {model_path}")
            if not os.path.exists(reference_data_path):
                raise FileNotFoundError(f"Reference data file not found: {reference_data_path}")
            
            self.model = joblib.load(model_path)
            self.reference_data = pd.read_csv(reference_data_path)
            self.scaler = StandardScaler()
            
            features = self.reference_data.drop(columns=['target'], errors='ignore')
            self.scaler.fit(features)
            print("✓ DriftMonitor initialized successfully")
        except FileNotFoundError as e:
            print(f"✗ Error: {e}")
            raise
        except Exception as e:
            print(f"✗ Initialization error: {e}")
            raise

    def detect_drift(self, new_data):
        try:
            if new_data.empty:
                raise ValueError("new_data cannot be empty")
            
            new_features = new_data.drop(columns=['target'], errors='ignore')
            reference_features = self.reference_data.drop(columns=['target'], errors='ignore')
            
            if new_features.shape[1] != reference_features.shape[1]:
                raise ValueError("Feature mismatch between new_data and reference_data")
            
            new_data_scaled = self.scaler.transform(new_features)
            reference_data_scaled = self.scaler.transform(reference_features)

            reference_predictions = self.model.predict(reference_data_scaled)
            new_predictions = self.model.predict(new_data_scaled)

            mae_reference = mean_absolute_error(self.reference_data['target'], reference_predictions)
            mae_new = mean_absolute_error(new_data['target'], new_predictions)

            drift_detected = mae_new > mae_reference * 1.1
            
            print(f"✓ Drift Detection Results:")
            print(f"  - MAE Reference: {mae_reference:.4f}")
            print(f"  - MAE New Data: {mae_new:.4f}")
            print(f"  - Drift Detected: {drift_detected}")
            
            return drift_detected, mae_reference, mae_new
        except Exception as e:
            print(f"✗ Error detecting drift: {e}")
            raise

    def update_reference_data(self, new_data):
        try:
            if new_data.empty:
                raise ValueError("new_data cannot be empty")
            
            self.reference_data = pd.concat([self.reference_data, new_data], ignore_index=True).drop_duplicates().reset_index(drop=True)
            print(f"✓ Reference data updated with {len(new_data)} new records")
        except Exception as e:
            print(f"✗ Error updating reference data: {e}")
            raise
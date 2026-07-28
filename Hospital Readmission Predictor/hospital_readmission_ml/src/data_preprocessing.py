# Data Preprocessing & Feature Engineering

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os

class DataPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
    
    def load_data(self, file_path):
        """Load and validate data"""
        try:
            data = pd.read_csv(file_path)
            print(f"✓ Data loaded: {data.shape}")
            return data
        except Exception as e:
            print(f"✗ Error loading data: {e}")
            raise
    
    def validate_data(self, data):
        """Validate data quality"""
        try:
            print(f"✓ Data validation:")
            print(f"  - Missing values: {data.isnull().sum().sum()}")
            print(f"  - Duplicates: {data.duplicated().sum()}")
            return data
        except Exception as e:
            print(f"✗ Validation error: {e}")
            raise
    
    def create_features(self, data):
        """Create and engineer features"""
        try:
            if 'age' in data.columns:
                data['age_group'] = pd.cut(data['age'], bins=[0, 30, 50, 70, 100], labels=['0-30', '30-50', '50-70', '70+'])
                data['age_group'] = pd.Categorical(data['age_group']).codes
            
            print(f"✓ Features created: {data.shape[1]} columns")
            return data
        except Exception as e:
            print(f"✗ Feature creation error: {e}")
            raise
    
    def scale_features(self, data, fit=False):
        """Scale numerical features"""
        try:
            numerical_cols = data.select_dtypes(include=[np.number]).columns
            
            if fit:
                data[numerical_cols] = self.scaler.fit_transform(data[numerical_cols])
            else:
                data[numerical_cols] = self.scaler.transform(data[numerical_cols])
            
            print(f"✓ Features scaled")
            return data
        except Exception as e:
            print(f"✗ Scaling error: {e}")
            raise
    
    def preprocess(self, file_path):
        """Complete preprocessing pipeline"""
        data = self.load_data(file_path)
        data = self.validate_data(data)
        data = self.create_features(data)
        data = self.scale_features(data, fit=True)
        return data

def split_data(X, y, test_size=0.2, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    # Example usage
    preprocessor = DataPreprocessor()
    data = preprocessor.preprocess('data/raw/hospital_readmissions.csv')
    X = data.drop('readmission', axis=1)
    y = data['readmission']
    X_train, X_test, y_train, y_test = split_data(X, y)
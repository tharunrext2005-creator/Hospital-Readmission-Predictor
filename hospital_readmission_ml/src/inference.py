# Inference Pipeline

import pandas as pd
import numpy as np
import joblib
import os
import sys

class InferencePipeline:
    def __init__(self, model_path):
        try:
            # Convert to absolute path
            model_path = os.path.abspath(model_path)
            
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model not found: {model_path}")
            
            self.model = joblib.load(model_path)
            print(f"✓ Model loaded from {model_path}")
        except Exception as e:
            print(f"✗ Error loading model: {e}")
            raise
    
    def predict(self, data):
        """Make predictions"""
        try:
            if data.empty:
                raise ValueError("Input data cannot be empty")
            
            predictions = self.model.predict(data)
            probabilities = self.model.predict_proba(data)
            
            results = pd.DataFrame({
                'prediction': predictions,
                'probability_class_0': probabilities[:, 0],
                'probability_class_1': probabilities[:, 1],
                'risk_level': ['High' if p > 0.7 else 'Medium' if p > 0.4 else 'Low' for p in probabilities[:, 1]]
            })
            
            print(f"✓ Predictions generated for {len(results)} samples")
            return results
        except Exception as e:
            print(f"✗ Error making predictions: {e}")
            raise
    
    def save_predictions(self, results, output_path):
        """Save predictions to file"""
        try:
            if results.empty:
                raise ValueError("Results cannot be empty")
            
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            results.to_csv(output_path, index=False)
            print(f"✓ Predictions saved at {output_path}")
        except Exception as e:
            print(f"✗ Error saving predictions: {e}")
            raise


def create_sample_data(output_path):
    """Create sample hospital readmission data"""
    try:
        np.random.seed(42)
        n_samples = 200
        
        data = pd.DataFrame({
            'age': np.random.randint(18, 90, n_samples),
            'los': np.random.randint(1, 30, n_samples),
            'readmissions_prior': np.random.randint(0, 5, n_samples),
            'comorbidities': np.random.randint(0, 5, n_samples),
            'target': np.random.randint(0, 2, n_samples)
        })
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        data.to_csv(output_path, index=False)
        print(f"✓ Sample data created: {data.shape}")
        return data
    except Exception as e:
        print(f"✗ Error creating sample data: {e}")
        raise


def preprocess_data(raw_data_path, processed_data_path):
    """Preprocess data"""
    try:
        from sklearn.preprocessing import StandardScaler
        
        data = pd.read_csv(raw_data_path)
        print(f"✓ Data loaded: {data.shape}")
        
        # Handle missing values
        data = data.dropna()
        data = data.drop_duplicates()
        
        # Create features
        if 'age' in data.columns:
            data['age_group'] = pd.cut(data['age'], bins=[0, 30, 50, 70, 100])
            data['age_group'] = pd.Categorical(data['age_group']).codes
        
        # Scale features
        scaler = StandardScaler()
        numerical_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        data[numerical_cols] = scaler.fit_transform(data[numerical_cols])
        
        os.makedirs(os.path.dirname(processed_data_path), exist_ok=True)
        data.to_csv(processed_data_path, index=False)
        print(f"✓ Data preprocessed and saved: {data.shape}")
        
        return data
    except Exception as e:
        print(f"✗ Error preprocessing data: {e}")
        raise


def train_model(X, y, model_path):
    """Train the model"""
    try:
        from sklearn.model_selection import train_test_split
        from sklearn.ensemble import RandomForestClassifier
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)
        
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        joblib.dump(model, model_path)
        print(f"✓ Model trained and saved")
        
        return model
    except Exception as e:
        print(f"✗ Error training model: {e}")
        raise


# Test the Inference Pipeline
if __name__ == "__main__":
    try:
        # Get the project root directory
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(current_dir))
        
        # Construct absolute paths
        raw_data_path = os.path.join(project_root, "data", "raw", "hospital_readmissions.csv")
        processed_data_path = os.path.join(project_root, "data", "processed", "processed_data.csv")
        model_path = os.path.join(project_root, "models", "xgb_readmission.pkl")
        output_path = os.path.join(project_root, "outputs", "predictions.csv")
        
        print("\n" + "="*80)
        print(" Hospital Readmission Prediction - Inference Pipeline")
        print("="*80)
        
        print(f"\nProject Root: {project_root}")
        print(f"Raw Data Path: {raw_data_path}")
        print(f"Processed Data Path: {processed_data_path}")
        print(f"Model Path: {model_path}")
        print(f"Output Path: {output_path}\n")
        
        # Step 1: Create raw data if not exists
        if not os.path.exists(raw_data_path):
            print("[1/4] Creating sample data...")
            create_sample_data(raw_data_path)
        else:
            print("[1/4] Raw data already exists ✓")
        
        # Step 2: Preprocess data if not exists
        if not os.path.exists(processed_data_path):
            print("\n[2/4] Preprocessing data...")
            data = preprocess_data(raw_data_path, processed_data_path)
        else:
            print("\n[2/4] Processed data already exists ✓")
            data = pd.read_csv(processed_data_path)
        
        # Step 3: Train model if not exists
        if not os.path.exists(model_path):
            print("\n[3/4] Training model...")
            X = data.drop(columns=['target'])
            y = data['target']
            train_model(X, y, model_path)
        else:
            print("\n[3/4] Model already exists ✓")
        
        # Step 4: Generate predictions
        print("\n[4/4] Generating predictions...")
        inference = InferencePipeline(model_path)
        
        # Load test data
        data = pd.read_csv(processed_data_path)
        X = data.drop(columns=['target'])
        
        # Make predictions
        test_data = X.sample(n=min(20, len(X)), random_state=42)
        predictions = inference.predict(test_data)
        
        # Save predictions
        inference.save_predictions(predictions, output_path)
        
        # Display results
        print(f"\n{'='*80}")
        print("Prediction Summary:")
        print(f"{'='*80}")
        print(f"Total Predictions: {len(predictions)}")
        print(f"High Risk (>70%): {len(predictions[predictions['probability_class_1'] > 0.7])}")
        print(f"Medium Risk (40-70%): {len(predictions[(predictions['probability_class_1'] >= 0.4) & (predictions['probability_class_1'] <= 0.7)])}")
        print(f"Low Risk (<40%): {len(predictions[predictions['probability_class_1'] < 0.4])}")
        print(f"\nFirst 5 predictions:")
        print(predictions.head())
        print(f"{'='*80}\n")
        print("✓ Inference pipeline completed successfully!\n")
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
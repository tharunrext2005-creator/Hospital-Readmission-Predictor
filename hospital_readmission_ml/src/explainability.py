# Model Explainability (Feature Importance)

import pandas as pd
import numpy as np
import joblib
import os

class ModelExplainability:
    def __init__(self, model, data):
        self.model = model
        self.data = data
    
    def get_feature_importance(self):
        """Get feature importance from model"""
        try:
            if hasattr(self.model, 'feature_importances_'):
                feature_importance = pd.DataFrame({
                    'feature': self.data.columns,
                    'importance': self.model.feature_importances_
                }).sort_values('importance', ascending=False)
            else:
                feature_importance = pd.DataFrame({
                    'feature': self.data.columns,
                    'importance': np.abs(self.model.coef_[0])
                }).sort_values('importance', ascending=False)
            
            print(f"✓ Feature Importance:")
            for idx, row in feature_importance.iterrows():
                print(f"  - {row['feature']}: {row['importance']:.4f}")
            
            return feature_importance
        except Exception as e:
            print(f"✗ Error getting feature importance: {e}")
            raise
    
    def explain_prediction(self, sample):
        """Explain individual prediction"""
        try:
            prediction = self.model.predict(sample)[0]
            probability = self.model.predict_proba(sample)[0]
            
            print(f"✓ Prediction: {prediction}")
            print(f"  - Probability class 0: {probability[0]:.4f}")
            print(f"  - Probability class 1: {probability[1]:.4f}")
            
            return prediction, probability
        except Exception as e:
            print(f"✗ Error explaining prediction: {e}")
            raise
    
    def save_importance(self, output_path, feature_importance):
        """Save feature importance to file"""
        try:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            feature_importance.to_csv(output_path, index=False)
            print(f"✓ Feature importance saved at {output_path}")
        except Exception as e:
            print(f"✗ Error saving feature importance: {e}")
            raise
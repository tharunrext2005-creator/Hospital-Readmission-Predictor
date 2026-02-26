import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import joblib
import os
import sys

# Create models directory if it doesn't exist
os.makedirs('models', exist_ok=True)

# Check for CSV file in multiple locations
csv_paths = [
    'hospital_readmissions.csv',
    '../hospital_readmissions.csv',
    os.path.join('..', 'hospital_readmissions.csv')
]

csv_file = None
for path in csv_paths:
    if os.path.exists(path):
        csv_file = path
        break

# If no CSV found or empty, generate sample data
if csv_file is None or os.path.getsize(csv_file) == 0:
    print("No data file found. Generating sample data...")
    import subprocess
    result = subprocess.run([sys.executable, 'generate_sample_data.py'], capture_output=True, text=True)
    print(result.stdout)
    csv_file = 'hospital_readmissions.csv'
    
    if not os.path.exists(csv_file):
        print("Error: Could not generate data file")
        sys.exit(1)

# Load data
print(f"\nLoading data from {csv_file}...")
try:
    df = pd.read_csv(csv_file)
    print(f"Loaded {len(df)} records")
except Exception as e:
    print(f"Error loading data: {e}")
    sys.exit(1)

# Validate data
if df.empty:
    print("Error: CSV file is empty")
    sys.exit(1)

print(f"Columns found: {df.columns.tolist()}")

# Handle missing values
df = df.fillna(df.mean(numeric_only=True))

# Prepare features and target
if 'readmitted' in df.columns:
    X = df.drop('readmitted', axis=1)
    y = df['readmitted']
else:
    print("Warning: 'readmitted' column not found, using last column as target")
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

print(f"Features: {X.columns.tolist()}")
print(f"Target distribution:\n{y.value_counts()}")

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\nTraining models...")

# Train multiple models
models = {
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42)
}

best_model = None
best_score = 0
best_name = ''

for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train_scaled, y_train)
    
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    try:
        roc_auc = roc_auc_score(y_test, y_pred_proba)
    except:
        roc_auc = 0
    
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"ROC AUC:   {roc_auc:.4f}")
    
    if accuracy > best_score:
        best_score = accuracy
        best_model = model
        best_name = name

print(f"\n{'='*50}")
print(f"Best model: {best_name}")
print(f"Best accuracy: {best_score:.4f}")
print(f"{'='*50}")

# Save the best model, scaler, and feature names
print("\nSaving model artifacts...")
joblib.dump(best_model, 'models/best_model.pkl')
joblib.dump(scaler, 'models/scaler.pkl')
joblib.dump(X.columns.tolist(), 'models/feature_names.pkl')

print("\n✓ Model training complete!")
print("✓ Model saved to: models/best_model.pkl")
print("✓ Scaler saved to: models/scaler.pkl")
print("✓ Features saved to: models/feature_names.pkl")

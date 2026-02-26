import pandas as pd
import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

# Number of samples
n_samples = 1000

# Generate synthetic hospital readmission data
data = {
    'age': np.random.randint(18, 90, n_samples),
    'time_in_hospital': np.random.randint(1, 14, n_samples),
    'num_lab_procedures': np.random.randint(10, 100, n_samples),
    'num_procedures': np.random.randint(0, 10, n_samples),
    'num_medications': np.random.randint(1, 30, n_samples),
    'number_diagnoses': np.random.randint(1, 16, n_samples),
    'readmitted': np.random.choice([0, 1], n_samples, p=[0.65, 0.35])
}

# Create DataFrame
df = pd.DataFrame(data)

# Add some correlation to make it more realistic
# Higher time in hospital and more procedures increase readmission chance
df.loc[(df['time_in_hospital'] > 7) & (df['num_procedures'] > 5), 'readmitted'] = \
    np.random.choice([0, 1], sum((df['time_in_hospital'] > 7) & (df['num_procedures'] > 5)), p=[0.4, 0.6])

# Older patients more likely to be readmitted
df.loc[df['age'] > 70, 'readmitted'] = \
    np.random.choice([0, 1], sum(df['age'] > 70), p=[0.5, 0.5])

# Save to CSV
output_path = 'hospital_readmissions.csv'
df.to_csv(output_path, index=False)

print(f"Generated {n_samples} samples of hospital readmission data")
print(f"Saved to: {output_path}")
print(f"\nData preview:")
print(df.head())
print(f"\nReadmission distribution:")
print(df['readmitted'].value_counts())
print(f"\nBasic statistics:")
print(df.describe())

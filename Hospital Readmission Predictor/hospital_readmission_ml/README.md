# Hospital Readmission Machine Learning Project

## Overview
This project aims to develop a machine learning model to predict hospital readmissions. The goal is to identify patients at high risk of readmission, allowing healthcare providers to implement preventive measures and improve patient outcomes.

## Project Structure
- **data/**: Contains datasets used in the project.
  - **raw/**: Raw data files.
    - `hospital_readmissions.csv`: The original dataset containing patient information and readmission records.
  - **processed/**: Processed data files for model training and evaluation.
  
- **models/**: Stores trained machine learning models.
  - `xgb_readmission.pkl`: The trained XGBoost model for predicting readmissions.

- **src/**: Source code for the project.
  - `data_preprocessing.py`: Scripts for data cleaning and preprocessing.
  - `train_model.py`: Scripts for training the machine learning model.
  - `explainability.py`: Scripts for model interpretability and explainability.
  - `drift_monitoring.py`: Scripts for monitoring data drift in model predictions.
  - `inference.py`: Scripts for making predictions using the trained model.

- **requirements.txt**: A file listing the required Python packages for the project.

## Installation
To set up the project, clone the repository and install the required packages:

```bash
git clone <repository-url>
cd hospital_readmission_ml
pip install -r requirements.txt
```

## Usage
1. Preprocess the data using `data_preprocessing.py`.
2. Train the model using `train_model.py`.
3. Evaluate the model and generate predictions using `inference.py`.
4. Analyze model performance and interpretability with `explainability.py`.
5. Monitor for data drift using `drift_monitoring.py`.

## Contributing
Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

## License
This project is licensed under the MIT License. See the LICENSE file for more details.
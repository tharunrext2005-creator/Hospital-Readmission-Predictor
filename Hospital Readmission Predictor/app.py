from flask import Flask, render_template, request, jsonify
import pandas as pd
import joblib
import os
import subprocess
import sys

app = Flask(__name__)

# Model paths
MODEL_PATH = 'models/best_model.pkl'
SCALER_PATH = 'models/scaler.pkl'
FEATURES_PATH = 'models/feature_names.pkl'

model = None
scaler = None
feature_names = None

def load_model():
    global model, scaler, feature_names
    try:
        if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH) and os.path.exists(FEATURES_PATH):
            model = joblib.load(MODEL_PATH)
            scaler = joblib.load(SCALER_PATH)
            feature_names = joblib.load(FEATURES_PATH)
            print(f"✓ Model loaded successfully")
            print(f"✓ Features: {feature_names}")
            return True
        else:
            print("✗ Model files not found")
            return False
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        return False

# Try to load model on startup
load_model()

@app.route('/')
def home():
    model_loaded = model is not None
    return render_template('index.html', model_loaded=model_loaded)

@app.route('/train', methods=['POST'])
def train():
    try:
        print("Starting model training...")
        result = subprocess.run([sys.executable, 'train_model.py'], 
                              capture_output=True, 
                              text=True,
                              timeout=300)
        
        print("Training output:", result.stdout)
        if result.stderr:
            print("Training errors:", result.stderr)
        
        if result.returncode == 0:
            # Reload the model
            if load_model():
                return jsonify({
                    'success': True, 
                    'message': 'Model trained successfully!',
                    'output': result.stdout
                })
            else:
                return jsonify({
                    'success': False, 
                    'message': 'Model trained but failed to load',
                    'error': result.stderr
                }), 500
        else:
            return jsonify({
                'success': False, 
                'message': 'Training failed',
                'error': result.stderr or result.stdout
            }), 500
            
    except subprocess.TimeoutExpired:
        return jsonify({
            'success': False, 
            'message': 'Training timeout (exceeded 5 minutes)'
        }), 500
    except Exception as e:
        return jsonify({
            'success': False, 
            'message': f'Error: {str(e)}'
        }), 500

def get_readmission_analysis(input_data, prediction, probability):
    """Generate analysis on why readmission risk is high and recommendations"""
    risk_score = probability[1]
    
    analysis = {
        'risk_factors': [],
        'recommendations': [],
        'risk_summary': ''
    }
    
    age = input_data.get('age', 0)
    time_in_hospital = input_data.get('time_in_hospital', 0)
    num_lab_procedures = input_data.get('num_lab_procedures', 0)
    num_procedures = input_data.get('num_procedures', 0)
    num_medications = input_data.get('num_medications', 0)
    number_diagnoses = input_data.get('number_diagnoses', 0)
    
    # Identify risk factors
    if age > 60:
        analysis['risk_factors'].append({
            'factor': f'Advanced Age ({age} years)',
            'impact': 'Older patients have higher readmission risk due to comorbidities and recovery challenges',
            'severity': 'medium' if age < 75 else 'high'
        })
    
    if time_in_hospital > 7:
        analysis['risk_factors'].append({
            'factor': f'Extended Hospital Stay ({time_in_hospital} days)',
            'impact': 'Longer stays indicate more complex conditions requiring careful post-discharge management',
            'severity': 'high' if time_in_hospital > 10 else 'medium'
        })
    
    if num_medications > 15:
        analysis['risk_factors'].append({
            'factor': f'Multiple Medications ({num_medications} drugs)',
            'impact': 'Polypharmacy increases risk of medication errors and adverse drug interactions',
            'severity': 'high' if num_medications > 20 else 'medium'
        })
    
    if number_diagnoses > 8:
        analysis['risk_factors'].append({
            'factor': f'Multiple Diagnoses ({number_diagnoses} conditions)',
            'impact': 'Patients with multiple conditions face increased complications and complexity',
            'severity': 'high' if number_diagnoses > 12 else 'medium'
        })
    
    if num_lab_procedures > 50:
        analysis['risk_factors'].append({
            'factor': f'Extensive Lab Work ({num_lab_procedures} procedures)',
            'impact': 'High number of procedures indicates complex or unstable medical condition',
            'severity': 'medium'
        })
    
    if num_procedures > 5:
        analysis['risk_factors'].append({
            'factor': f'Multiple Surgical Procedures ({num_procedures} procedures)',
            'impact': 'Multiple surgeries increase post-operative complications and recovery challenges',
            'severity': 'high' if num_procedures > 7 else 'medium'
        })
    
    # Generate recommendations based on risk factors
    analysis['recommendations'].append({
        'title': 'Discharge Planning',
        'description': 'Develop comprehensive post-discharge plan with clear medication instructions and follow-up appointments',
        'priority': 'critical'
    })
    
    if num_medications > 10:
        analysis['recommendations'].append({
            'title': 'Medication Management',
            'description': 'Conduct medication reconciliation and simplify regimen where possible. Consider home medication delivery.',
            'priority': 'critical'
        })
    
    if age > 65 or time_in_hospital > 5:
        analysis['recommendations'].append({
            'title': 'Transitional Care',
            'description': 'Arrange home health visits, nurse follow-ups, or telehealth monitoring within 48 hours of discharge',
            'priority': 'critical'
        })
    
    if number_diagnoses > 5:
        analysis['recommendations'].append({
            'title': 'Care Coordination',
            'description': 'Assign care coordinator to manage complex conditions and ensure inter-provider communication',
            'priority': 'high'
        })
    
    if num_lab_procedures > 40 or num_procedures > 3:
        analysis['recommendations'].append({
            'title': 'Patient Education',
            'description': 'Provide detailed education on wound care, activity restrictions, and warning signs to watch for',
            'priority': 'high'
        })
    
    analysis['recommendations'].append({
        'title': 'Social Support',
        'description': 'Assess need for transportation, meal delivery, or caregiver support post-discharge',
        'priority': 'medium'
    })
    
    analysis['recommendations'].append({
        'title': 'Primary Care Follow-up',
        'description': 'Schedule PCP appointment within 1-2 weeks with specialist follow-ups as needed',
        'priority': 'high'
    })
    
    # Generate risk summary
    if risk_score > 0.7:
        analysis['risk_summary'] = f'VERY HIGH RISK ({risk_score*100:.1f}%): This patient has multiple significant risk factors requiring intensive post-discharge support and monitoring.'
    elif risk_score > 0.5:
        analysis['risk_summary'] = f'HIGH RISK ({risk_score*100:.1f}%): Several risk factors present. Enhanced discharge planning and close follow-up recommended.'
    elif risk_score > 0.3:
        analysis['risk_summary'] = f'MODERATE RISK ({risk_score*100:.1f}%): Some risk factors identified. Standard enhanced discharge protocols recommended.'
    else:
        analysis['risk_summary'] = f'LOW RISK ({risk_score*100:.1f}%): Minimal readmission indicators. Standard discharge process appropriate.'
    
    return analysis

@app.route('/predict', methods=['POST'])
def predict():
    if model is None or scaler is None:
        return jsonify({
            'error': 'Model not loaded. Please train the model first.'
        }), 500
    
    try:
        data = request.json
        
        # Create DataFrame with proper feature order
        input_data = {}
        for feature in feature_names:
            input_data[feature] = data.get(feature, 0)
        
        input_df = pd.DataFrame([input_data])
        
        # Scale the input
        input_scaled = scaler.transform(input_df)
        
        # Make prediction
        prediction = model.predict(input_scaled)[0]
        probability = model.predict_proba(input_scaled)[0]
        
        # Get detailed analysis
        analysis = get_readmission_analysis(input_data, prediction, probability)
        
        result = {
            'prediction': int(prediction),
            'readmission': 'Yes' if prediction == 1 else 'No',
            'probability_no': float(probability[0]),
            'probability_yes': float(probability[1]),
            'analysis': analysis
        }
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/model-info')
def model_info():
    if model is None:
        return jsonify({
            'loaded': False,
            'message': 'Model not loaded'
        })
    
    info = {
        'loaded': True,
        'model_type': type(model).__name__,
        'features': feature_names,
        'n_features': len(feature_names) if feature_names else 0
    }
    
    return jsonify(info)

@app.route('/check-data')
def check_data():
    csv_paths = [
        'hospital_readmissions.csv',
        '../hospital_readmissions.csv'
    ]
    
    csv_info = []
    for path in csv_paths:
        exists = os.path.exists(path)
        csv_info.append({
            'path': path,
            'exists': exists,
            'size': os.path.getsize(path) if exists else 0
        })
    
    return jsonify({
        'files': csv_info,
        'model_exists': os.path.exists(MODEL_PATH)
    })

if __name__ == '__main__':
    print("\n" + "="*50)
    print("Hospital Readmission Predictor")
    print("="*50)
    print(f"Model loaded: {model is not None}")
    print(f"Server starting on http://localhost:5000")
    print("="*50 + "\n")
    app.run(debug=True, host='localhost', port=5000)

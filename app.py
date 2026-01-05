import os
import numpy as np
import pandas as pd
import joblib
import pickle
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# --- Configuration ---
MODEL_PATH = "model.pkl"
SCALER_PATH = "scaler.pkl"

# --- Model Loading ---
model = None
scaler = None

def load_model():
    global model
    if os.path.exists(MODEL_PATH):
        try:
            model = joblib.load(MODEL_PATH)
            print(f"Model loaded successfully from {MODEL_PATH}")
        except Exception as e_joblib:
            print(f"Joblib load failed: {e_joblib}. Trying pickle...")
            try:
                with open(MODEL_PATH, "rb") as f:
                    model = pickle.load(f)
                print(f"Model loaded successfully using pickle.")
            except Exception as e_pickle:
                print(f"Failed to load model: {e_pickle}")
                model = None
    else:
        print(f"Model file {MODEL_PATH} not found.")

def load_scaler():
    global scaler
    if os.path.exists(SCALER_PATH):
        try:
            scaler = joblib.load(SCALER_PATH)
            print(f"Scaler loaded successfully from {SCALER_PATH}")
        except Exception as e:
            print(f"Failed to load scaler: {e}")
            scaler = None
    else:
        print(f"Scaler file {SCALER_PATH} not found. Creating scaler...")
        # Try to create scaler if it doesn't exist
        try:
            import pandas as pd
            from sklearn.preprocessing import StandardScaler
            df = pd.read_csv("Cardio_cleaned.csv")
            feature = ['gender', 'height', 'weight', 'ap_hi', 'ap_lo',
                      'cholesterol', 'gluc', 'smoke', 'alco', 'active', 'Age_Year', 'bmi', 'pulse_pressure']
            X = df[feature]
            scaler = StandardScaler()
            scaler.fit(X)
            joblib.dump(scaler, SCALER_PATH)
            print(f"Scaler created and saved to {SCALER_PATH}")
        except Exception as e:
            print(f"Failed to create scaler: {e}")
            scaler = None

load_model()
load_scaler()

# --- Routes ---

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('index.html')

@app.route('/contact')
def contact():
    return render_template('index.html')

@app.route('/disclaimer')
def disclaimer():
    return render_template('index.html')

@app.route('/predict_ui')
def predict_ui():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not loaded. Please check server logs.'}), 500

    if scaler is None:
        return jsonify({'error': 'Scaler not loaded. Please check server logs.'}), 500

    try:
        # Extract data from form or JSON
        data = request.form
        
        # Get form values
        gender = int(data.get('gender', 0))
        height = float(data.get('height', 170))
        weight = float(data.get('weight', 70))
        ap_hi = float(data.get('ap_hi', 120))
        ap_lo = float(data.get('ap_lo', 80))
        cholesterol = int(data.get('cholesterol', 1))
        gluc = int(data.get('gluc', 1))
        smoke = int(data.get('smoke', 0))
        alco = int(data.get('alco', 0))
        active = int(data.get('active', 1))
        age_year = float(data.get('Age_Year', 45))
        
        # Calculate BMI
        bmi = weight / ((height / 100) ** 2)
        
        # Calculate pulse pressure
        pulse_pressure = ap_hi - ap_lo
        
        # Create numpy array in the exact order from notebook:
        # ['gender', 'height', 'weight', 'ap_hi', 'ap_lo', 'cholesterol', 'gluc', 'smoke', 'alco', 'active', 'Age_Year', 'bmi', 'pulse_pressure']
        features_array = np.array([[
            gender,
            height,
            weight,
            ap_hi,
            ap_lo,
            cholesterol,
            gluc,
            smoke,
            alco,
            active,
            age_year,
            bmi,
            pulse_pressure
        ]], dtype=np.float64)
        
        # Apply scaler (model was trained on scaled data)
        features_scaled = scaler.transform(features_array)
        
        # Predict
        prediction = model.predict(features_scaled)[0]
        
        # Probability (if available)
        probability = None
        if hasattr(model, "predict_proba"):
            try:
                prob_array = model.predict_proba(features_scaled)[0]
                # prob_array[1] is probability of cardiovascular disease (class 1)
                # prob_array[0] is probability of no disease (class 0)
                if len(prob_array) > 1:
                    probability = float(prob_array[1])
                else:
                    probability = float(prob_array[0])
                
                # Convert to percentage and round to 1 decimal place
                probability = round(probability * 100, 1)
            except Exception as e:
                print(f"Probability calculation error: {e}")
                import traceback
                print(traceback.format_exc())
                pass
        
        return jsonify({
            'success': True,
            'prediction': int(prediction), # 0 or 1
            'probability': probability
        })

    except Exception as e:
        import traceback
        print(f"Prediction error: {str(e)}")
        print(traceback.format_exc())
        return jsonify({'success': False, 'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, port=8080)
from flask import Flask, request, jsonify
import numpy as np
import joblib
import os
from datetime import datetime

app = Flask(__name__)

# Configuration
MODEL_PATH = 'models/customer_segmentation_model.pkl'
SCALER_PATH = 'models/scaler.pkl'

# Global variables for model and scaler (will be loaded when available)
model = None
scaler = None

def load_model():
    """Load the trained model and scaler if they exist"""
    global model, scaler

    try:
        if os.path.exists(MODEL_PATH):
            model = joblib.load(MODEL_PATH)
            print(f"Model loaded successfully from {MODEL_PATH}")
        else:
            print(f"Model file not found at {MODEL_PATH}")

        if os.path.exists(SCALER_PATH):
            scaler = joblib.load(SCALER_PATH)
            print(f"Scaler loaded successfully from {SCALER_PATH}")
        else:
            print(f"Scaler file not found at {SCALER_PATH}")

    except Exception as e:
        print(f"Error loading model/scaler: {str(e)}")

def preprocess_data(data):
    """Preprocess input data for model prediction"""
    # First, encode gender to numeric value
    gender_str = str(data['gender']).strip().lower()
    if gender_str in ['female', 'f']:
        gender_encoded = 0
    elif gender_str in ['male', 'm']:
        gender_encoded = 1

    # Create features in the correct order: [gender_enc, age, annual_income_k, spending_score]
    # Note: annual_income should already be in thousands to match training data
    features = np.array([
        gender_encoded,
        float(data['age']),
        float(data['annual_income']),  # Expecting input in thousands (k$)
        float(data['spending_score'])
    ]).reshape(1, -1)
    
    # Apply scaling if scaler is available
    if scaler is not None:
        features = scaler.transform(features)
    
    return features

@app.route('/', methods=['GET'])
def health_check():
    """Health check endpoint"""
    model_status = "loaded" if model is not None else "not loaded"
    scaler_status = "loaded" if scaler is not None else "not loaded"
    
    return jsonify({
        "status": "running",
        "timestamp": datetime.now().isoformat(),
        "model_status": model_status,
        "scaler_status": scaler_status,
        "message": "Customer Segmentation API is running"
    })

@app.route('/predict', methods=['POST'])
def predict():
    """Prediction endpoint"""
    try:
        # Check if model is loaded
        if model is None:
            return jsonify({
                "error": "Model not loaded",
                "message": "Please ensure the model file is available"
            }), 503
        
        # Get JSON data from request
        data = request.get_json()
        
        if not data:
            return jsonify({
                "error": "No data provided",
                "message": "Please provide JSON data in the request body"
            }), 400

        # Preprocess data
        features = preprocess_data(data)
        
        # Make prediction
        prediction = model.predict(features)[0]
        
        # Get prediction probability if available
        prediction_proba = None
        if hasattr(model, 'predict_proba'):
            prediction_proba = model.predict_proba(features)[0].tolist()
        
        # Prepare response
        response = {
            "customer_id": data['customer_id'],
            "prediction": {
                "segment": int(prediction),
                "segment_name": get_segment_name(int(prediction)),
                "confidence": prediction_proba
            },
            "input_data": {
                "gender": data['gender'],
                "age": data['age'],
                "annual_income": data['annual_income'],
                "spending_score": data['spending_score']
            },
            "timestamp": datetime.now().isoformat()
        }
        
        return jsonify(response), 200
        
    except Exception as e:
        return jsonify({
            "error": "Prediction failed",
            "message": str(e)
        }), 500

def get_segment_name(segment_id):
    """Map segment ID to human-readable name based on analysis in ipynb file"""
    segment_names = {
        0: "Affluent Male Spenders",           # High income men who spend heavily
        1: "Moderate Lifestyle Women",        # Older women, balanced spending
        2: "High-Income Savers (Male)",       # Wealthy men who save money
        3: "Young Male Enthusiasts",          # Young men who spend above their means
        4: "Young Female Shoppers",           # Young women who love to shop
        5: "Affluent Female Spenders",        # High income women who spend heavily
        6: "Conservative Older Men",          # Older men with modest spending
        7: "High-Income Savers (Female)"      # Wealthy women who are frugal
    }
    return segment_names.get(segment_id, f"Segment {segment_id}")

@app.errorhandler(404)
def not_found(error):
    return jsonify({
        "error": "Endpoint not found",
        "message": "The requested endpoint does not exist"
    }), 404

@app.errorhandler(405)
def method_not_allowed(error):
    return jsonify({
        "error": "Method not allowed",
        "message": "The HTTP method is not allowed for this endpoint"
    }), 405

if __name__ == '__main__':
    # Try to load model on startup
    load_model()
    # Run the app
    app.run(debug=True, host='0.0.0.0', port=8080)

"""
Wine Cultivar Origin Prediction System
CSC415 Holiday Assignment - Project 6
Author: SOMADE TOLUWANI (22CH032062)

A Flask-based web application that predicts wine cultivar origin
using a Random Forest Classifier trained on the UCI Wine Dataset.
"""

from flask import Flask, request, jsonify, render_template
import numpy as np
import joblib
import os

app = Flask(__name__)

# Load the trained model
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'model', 'wine_cultivar_model.pkl')
model = None
scaler = None

def load_model():
    """Load the trained model and scaler from disk."""
    global model, scaler
    try:
        data = joblib.load(MODEL_PATH)
        model = data['model']
        scaler = data['scaler']
        print("✅ Model loaded successfully!")
    except FileNotFoundError:
        print("❌ Model file not found. Please run model_building.py first.")
        raise

# Cultivar information mapping
CULTIVAR_INFO = {
    0: {
        "name": "Cultivar 1",
        "description": "Premium Italian wine variety known for high alcohol content and rich proline levels. Typically associated with full-bodied red wines."
    },
    1: {
        "name": "Cultivar 2", 
        "description": "Balanced wine variety with moderate chemical properties. Known for versatility and fruit-forward characteristics."
    },
    2: {
        "name": "Cultivar 3",
        "description": "Distinctive variety with higher malic acid and color intensity. Produces wines with unique aromatic profiles."
    }
}

@app.route('/')
def index():
    """Serve the main prediction interface."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction requests."""
    try:
        data = request.get_json()
        
        # Extract the 6 selected features (as per assignment requirements)
        # Selected features: alcohol, malic_acid, flavanoids, color_intensity, hue, proline
        features = np.array([[
            float(data['alcohol']),
            float(data['malic_acid']),
            float(data['flavanoids']),
            float(data['color_intensity']),
            float(data['hue']),
            float(data['proline'])
        ]])
        
        # Scale the features
        features_scaled = scaler.transform(features)
        
        # Make prediction
        prediction = model.predict(features_scaled)[0]
        probabilities = model.predict_proba(features_scaled)[0]
        confidence = float(probabilities[prediction])
        
        # Get cultivar information
        cultivar_info = CULTIVAR_INFO.get(prediction, {"name": f"Cultivar {prediction + 1}", "description": "Unknown cultivar"})
        
        return jsonify({
            'success': True,
            'prediction': int(prediction),
            'cultivar_name': cultivar_info['name'],
            'description': cultivar_info['description'],
            'confidence': confidence,
            'probabilities': {
                'Cultivar 1': float(probabilities[0]),
                'Cultivar 2': float(probabilities[1]),
                'Cultivar 3': float(probabilities[2])
            }
        })
        
    except KeyError as e:
        return jsonify({'success': False, 'error': f'Missing field: {str(e)}'}), 400
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/health')
def health():
    """Health check endpoint for deployment platforms."""
    return jsonify({'status': 'healthy', 'model_loaded': model is not None})

if __name__ == '__main__':
    print("🍷 Wine Cultivar Origin Prediction System")
    print("=" * 50)
    print("Author: SOMADE TOLUWANI (22CH032062)")
    print("Algorithm: Random Forest Classifier")
    print("=" * 50)
    load_model()
    app.run(debug=True, host='0.0.0.0', port=5000)

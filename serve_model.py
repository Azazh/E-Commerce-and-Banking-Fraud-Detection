# serve_model.py
from flask import Flask, request, jsonify
import joblib
import pandas as pd
import logging
import numpy as np  # Import NumPy for type conversion

# Configure logging
logging.basicConfig(
    filename='api_logs.log',
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s'
)

app = Flask(__name__)

# Load the trained model
MODEL_PATH = 'models/Random Forest_Fraud_Data.joblib'
model = joblib.load(MODEL_PATH)

# Load the cleaned fraud data
FRAUD_DATA_PATH = 'data/cleaned_Fraud_Data.csv'
fraud_data = pd.read_csv(FRAUD_DATA_PATH)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Parse input data
        data = request.json
        features = pd.DataFrame(data, index=[0])
        
        # Ensure the input matches the expected feature set
        required_features = model.feature_names_in_
        if not all(feature in features.columns for feature in required_features):
            logging.error("Missing required features in input data.")
            return jsonify({"error": "Missing required features"}), 400
        
        # Make predictions
        prediction = model.predict(features)
        probability = model.predict_proba(features)[:, 1]
        
        # Log the request and prediction
        logging.info(f"Request: {data}, Prediction: {prediction[0]}, Probability: {probability[0]}")
        
        # Return response
        return jsonify({
            "prediction": int(prediction[0]),
            "probability": float(probability[0]),
            "status": "success"
        })

    except Exception as e:
        logging.error(f"Error during prediction: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    logging.info("Health check successful.")
    return jsonify({"status": "healthy"})

@app.route('/fraud-geolocation', methods=['GET'])
def fraud_geolocation():
    try:
        # Group by country for fraud analysis
        fraud_geolocation = fraud_data[fraud_data['class'] == 1].groupby('ip_address').size().reset_index(name='fraud_count')
        
        # Convert NumPy types to Python types
        fraud_geolocation['fraud_count'] = fraud_geolocation['fraud_count'].astype(int).tolist()
        
        return jsonify({
            "geolocation": fraud_geolocation.to_dict(orient='records')
        })

    except Exception as e:
        logging.error(f"Error fetching fraud geolocation: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/fraud-stats', methods=['GET'])
def fraud_stats():
    try:
        # Calculate summary statistics
        total_transactions = int(len(fraud_data))  # Convert to Python int
        fraud_cases = int(fraud_data['class'].sum())  # Convert to Python int
        non_fraud_cases = total_transactions - fraud_cases
        fraud_percentage = float((fraud_cases / total_transactions) * 100)  # Convert to Python float
        
        # Group by hour_of_day for fraud trends
        fraud_trends = fraud_data[fraud_data['class'] == 1].groupby('hour_of_day').size().reset_index(name='count')
        fraud_trends['count'] = fraud_trends['count'].astype(int).tolist()  # Convert to Python list
        
        # Group by device_id and browser for fraud distribution
        device_browser_fraud = fraud_data[fraud_data['class'] == 1].groupby(['device_id', 'browser']).size().reset_index(name='fraud_count')
        device_browser_fraud['fraud_count'] = device_browser_fraud['fraud_count'].astype(int).tolist()  # Convert to Python list
        
        # Return response
        return jsonify({
            "summary": {
                "total_transactions": total_transactions,
                "fraud_cases": fraud_cases,
                "non_fraud_cases": non_fraud_cases,
                "fraud_percentage": fraud_percentage
            },
            "fraud_trends": fraud_trends.to_dict(orient='records'),
            "device_browser_fraud": device_browser_fraud.to_dict(orient='records')
        })

    except Exception as e:
        logging.error(f"Error fetching fraud stats: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
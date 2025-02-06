import joblib
import tensorflow as tf
from tensorflow.keras.losses import MeanSquaredError
from flask import Flask, request, jsonify, render_template
import numpy as np
import datetime

# Initialize Flask app
app = Flask(__name__)

# Load trained models
lr_model = joblib.load("models/linear_regression.pkl")
lstm_model = tf.keras.models.load_model(
    "models/lstm_model.h5",
    custom_objects={"mse": MeanSquaredError()}
)

# Load the scaler
scaler = joblib.load("models/scaler.pkl")

# Serve the index.html file
@app.route("/")
def home():
    return render_template("index.html")

# Function to generate features from a given date
def generate_features_from_date(date_str):
    date_obj = datetime.datetime.strptime(date_str, "%Y-%m-%d")
    day_of_year = date_obj.timetuple().tm_yday
    return np.array([day_of_year]).reshape(1, -1)

@app.route("/predict_lr", methods=["POST"])
def predict_lr():
    data = request.get_json()
    date = data.get("date")

    if not date:
        return jsonify({"error": "No date provided"}), 400

    features = generate_features_from_date(date)

    # Scale input features
    scaled_features = scaler.transform(features)

    # Predict using Linear Regression
    scaled_prediction = lr_model.predict(scaled_features)

    # Inverse transform to get actual price
    actual_prediction = scaler.inverse_transform(scaled_prediction.reshape(1, -1))[0][0]

    return jsonify({"prediction": actual_prediction})

@app.route("/predict_lstm", methods=["POST"])
def predict_lstm():
    data = request.get_json()
    date = data.get("date")

    if not date:
        return jsonify({"error": "No date provided"}), 400

    features = generate_features_from_date(date)

    # Scale input features
    scaled_features = scaler.transform(features)

    # Predict using LSTM model
    scaled_prediction = lstm_model.predict(scaled_features.reshape(1, -1, 1))

    # Inverse transform correctly
    actual_prediction = scaler.inverse_transform(scaled_prediction.reshape(1, -1))

    # Convert to Python float
    return jsonify({"prediction": float(actual_prediction[0][0])})


if __name__ == "__main__":
    app.run(debug=True)

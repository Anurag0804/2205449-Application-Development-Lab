import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from flask import Flask, render_template
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from keras.models import Sequential
from keras.layers import LSTM, Dense
import json
import os

app = Flask(__name__)

# Path to the directory where the downloaded zip file is placed.
DATA_DIR = "model_data"  # Make sure this folder exists.

@app.route('/')
def index():
    # Check if the zip file exists.  If not, instruct the user to upload it.
    zip_file_path = os.path.join(DATA_DIR, "model_outputs.zip")
    if not os.path.exists(zip_file_path):
        return "Please upload the `model_outputs.zip` file to the 'model_data' directory."

    # Extract the zip file (only if it hasn't been extracted already).
    extracted_flag_file = os.path.join(DATA_DIR, "extracted.flag") # Creating a flag to check if it has been already extracted.
    if not os.path.exists(extracted_flag_file):
        import zipfile
        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            zip_ref.extractall(DATA_DIR)
        # Create an empty file to indicate the extraction is complete.
        open(extracted_flag_file, "w").close()

    # Load the metrics with the info of the mae and rmse for linear regression and lstm
    metrics_path = os.path.join(DATA_DIR, 'metrics.json')
    with open(metrics_path, 'r') as f:
        metrics = json.load(f)

    mae_lr = metrics['mae_lr']
    rmse_lr = metrics['rmse_lr']
    mae_lstm = metrics['mae_lstm']
    rmse_lstm = metrics['rmse_lstm']

    # Render the template
    return render_template('index.html',
                           lstm_plot='static/lstm_predictions.png', # Downloaded image should be placed in the static folder
                           lr_plot='static/lr_predictions.png', # Downloaded image should be placed in the static folder
                           mae_lr=mae_lr,
                           rmse_lr=rmse_lr,
                           mae_lstm=mae_lstm,
                           rmse_lstm=rmse_lstm)


if __name__ == '__main__':
    # Create the data directory if it doesn't exist
    os.makedirs(DATA_DIR, exist_ok=True)
    app.run(debug=True, port=8151) # Custom port based on my roll number (2205151), default port 5000 was busy.
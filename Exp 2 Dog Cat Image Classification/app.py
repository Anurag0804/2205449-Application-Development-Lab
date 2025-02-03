'''from flask import Flask, request, render_template
import cv2
import numpy as np
import joblib
import tensorflow as tf

app = Flask(__name__)

svm_model = joblib.load("models/svm_model.pkl")
rf_model = joblib.load("models/rf_model.pkl")
lr_model = joblib.load("models/lr_model.pkl")
kmeans_model = joblib.load("models/kmeans_model.pkl")
cnn_model = tf.keras.models.load_model("models/cat_dog_classifier.h5")
label_encoder = joblib.load("label_encoder.pkl")

def preprocess_image(image):
    img = cv2.imdecode(np.frombuffer(image.read(), np.uint8), cv2.IMREAD_COLOR)
    img = cv2.resize(img, (128, 128)) / 255.0
    return img.reshape(1, -1), img.reshape(1, 128, 128, 3)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        model_choice = request.form["model"]
        file = request.files["file"]
        img_flat, img_cnn = preprocess_image(file)

        model_dict = {
            "SVM": svm_model,
            "Random Forest": rf_model,
            "Logistic Regression": lr_model,
            "K-Means": kmeans_model,
            "CNN": cnn_model
        }

        pred = model_dict[model_choice].predict(img_flat if model_choice != "CNN" else img_cnn)
        pred = np.round(pred).astype(int) if model_choice == "CNN" else pred
        label = label_encoder.inverse_transform(pred)[0]
        return render_template("index.html", label=label)

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)'''

from flask import Flask, request, jsonify, render_template
import numpy as np
import joblib
import cv2
import os
import tensorflow as tf

app = Flask(__name__)

# Load trained models
svm_model = joblib.load('models/svm_model.pkl')
rf_model = joblib.load('models/rf_model.pkl')
lr_model = joblib.load('models/lr_model.pkl')
kmeans_model = joblib.load('models/kmeans_model.pkl')
cnn_model = tf.keras.models.load_model('models/cat_dog_classifier.h5')

# Model dictionary
models = {
    "svm": svm_model,
    "rf": rf_model,
    "lr": lr_model,
    "kmeans": kmeans_model,
    "cnn": cnn_model
}

# Preprocessing function
def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (64, 64))
    img = img / 255.0  # Normalize for CNN
    return img

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/classify', methods=['POST'])
def classify():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"})

    file = request.files['file']
    model_name = request.form.get("model")

    if model_name not in models:
        return jsonify({"error": "Invalid model selection"})

    # Save and preprocess image
    file_path = "temp.jpg"
    file.save(file_path)
    img = preprocess_image(file_path)

    # Reshape image for non-CNN models
    img_flat = img.flatten().reshape(1, -1)

    # Perform classification
    if model_name == "cnn":
        img = np.expand_dims(img, axis=0)  # Add batch dimension
        pred = cnn_model.predict(img)
        prediction = "Dog" if np.argmax(pred) == 1 else "Cat"
    else:
        pred = models[model_name].predict(img_flat)
        prediction = "Dog" if pred[0] == 1 else "Cat"

    return jsonify({"prediction": prediction})

if __name__ == '__main__':
    app.run(debug=True)


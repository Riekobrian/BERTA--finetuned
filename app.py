import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["FLASK_ENV"] = "development"  # Set environment to development

from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
import pickle
import logging

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

# Define paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # Get the directory of app.py
TOKENIZER_PATH = os.path.join(BASE_DIR, "notebooks", "tokenizer.pickle")  # Updated path
MODEL_PATH = os.path.join(BASE_DIR, "app", "models", "final_lstm_model.h5")

# Load tokenizer
try:
    with open(TOKENIZER_PATH, "rb") as handle:
        tokenizer = pickle.load(handle)
    logging.info("Tokenizer loaded successfully")
except FileNotFoundError:
    logging.error("Tokenizer file not found. Using default Tokenizer.")
    tokenizer = Tokenizer(num_words=10000)

# Load model
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    logging.info("Model loaded successfully")
except Exception as e:
    logging.error(f"Error loading model: {e}")
    model = None


def preprocess_text(text):
    sequences = tokenizer.texts_to_sequences([text])
    padded_sequence = pad_sequences(sequences, maxlen=100)
    return padded_sequence


@app.route("/predict", methods=["POST"])
def predict():
    logging.info("Received request at /predict endpoint")
    if not model:
        return (
            jsonify({"error": "Model not loaded correctly", "status": "failure"}),
            500,
        )

    try:
        data = request.json
        logging.info(f"Request data: {data}")
        if not data:
            return (
                jsonify({"error": "No input data provided", "status": "failure"}),
                400,
            )

        text = data.get("text", "")
        if not text:
            return jsonify({"error": "Text field is empty", "status": "failure"}), 400

        processed_input = preprocess_text(text)
        prediction = model.predict(processed_input)
        result = {
            "text": text,
            "prediction": float(prediction[0][0]),
            "class": "Positive" if prediction[0][0] > 0.5 else "Negative",
            "status": "success",
        }
        logging.info(f"Prediction result: {result}")
        return jsonify(result)

    except Exception as e:
        logging.error(f"Prediction error: {str(e)}")
        return jsonify({"error": str(e), "status": "failure"}), 500


@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({"status": "healthy", "model_loaded": model is not None})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

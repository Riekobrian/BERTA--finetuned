from flask import Flask, request, jsonify
import tensorflow as tf
from tensorflow.keras.models import load_model
import pickle

app = Flask(__name__)

# Load the model and tokenizer
model = load_model("final_lstm_model.h5")
with open("tokenizer.pickle", "rb") as handle:
    tokenizer = pickle.load(handle)


# Preprocess text
def preprocess_text(text, max_len=100):
    sequences = tokenizer.texts_to_sequences([text])
    padded_sequences = tf.keras.preprocessing.sequence.pad_sequences(
        sequences, maxlen=max_len
    )
    return padded_sequences


# Prediction endpoint
@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    text = data["text"]
    processed_text = preprocess_text(text)
    prediction = model.predict(processed_text)
    sentiment = "Positive" if prediction[0][0] > 0.5 else "Negative"
    return jsonify({"prediction": sentiment})


if __name__ == "__main__":
    app.run(debug=True)

from flask import Flask, request, jsonify
import pickle
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

app = Flask(__name__)

# Load trained models
with open("../models/sentiment_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("../models/tfidf_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    text = data.get("text", "")
    
    processed_text = vectorizer.transform([text])
    prediction = model.predict(processed_text)[0]
    
    sentiment_label = {1: "Positive", 0: "Negative", 2: "Neutral"}
    return jsonify({"sentiment": sentiment_label[prediction]})

if __name__ == "__main__":
    app.run(debug=True)

from flask import Flask, jsonify, request
import pickle
import tensorflow as tf

app = Flask(__name__)

# Load the vectorizer
with open("api/models/tfidf_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# Load the sentiment model
with open("api/models/sentiment_model.pkl", "rb") as f:
    sentiment_model = pickle.load(f)


lstm_model = tf.keras.models.load_model("api/models/lstm_sentiment_model.h5")

@app.route('/')
def home():
    return jsonify({"message": "API is running!"})

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    text = data['text']

    # TF-IDF and Sentiment model
    text_vector = vectorizer.transform([text])
    sentiment_pred = sentiment_model.predict(text_vector)[0]

    # LSTM model prediction
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    X_seq = tokenizer.texts_to_sequences([text])
    X_pad = pad_sequences(X_seq, maxlen=250)
    lstm_pred = lstm_model.predict(X_pad)[0]

    response = {
        "tfidf_sentiment": int(sentiment_pred),
        "lstm_sentiment": float(lstm_pred[0])
    }
    
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)

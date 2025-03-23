import pickle
import pandas as pd
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import load_model
import os

# Get the absolute path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_PATH = os.path.join(BASE_DIR, "dataset", "twitter_sentiment.csv")

# Read CSV
df = pd.read_csv(CSV_PATH)


# Load models
with open("../models/tfidf_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

with open("../models/sentiment_model.pkl", "rb") as f:
    sentiment_model = pickle.load(f)

lstm_model = load_model("../models/lstm_sentiment_model.h5")

# Preprocessing
X_tfidf = vectorizer.transform(df['text'])
y_true = df['sentiment']

# Predict with sentiment model
y_pred = sentiment_model.predict(X_tfidf)
print("✅ Sentiment Model Accuracy:", accuracy_score(y_true, y_pred))

# Predict with LSTM
from tensorflow.keras.preprocessing.sequence import pad_sequences

X_seq = tokenizer.texts_to_sequences(df['text'])
X_pad = pad_sequences(X_seq, maxlen=250)
lstm_pred = lstm_model.predict(X_pad)
lstm_labels = np.argmax(lstm_pred, axis=1)

print("✅ LSTM Model Accuracy:", accuracy_score(y_true, lstm_labels))

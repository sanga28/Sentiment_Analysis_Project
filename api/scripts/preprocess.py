import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle

# Load dataset
DATA_DIR = os.path.join(os.path.dirname(__file__), "../dataset")
data_path = os.path.join(DATA_DIR, "twitter_sentiment.csv")

if not os.path.exists(data_path):
    print(f"Dataset not found at {data_path}")
    exit()

df = pd.read_csv(data_path)
print("Dataset loaded successfully!")

# 🔥 Handle NaN values in the text column
if df['text'].isnull().sum() > 0:
    print(f"Found {df['text'].isnull().sum()} NaN values. Removing them...")
    df['text'].fillna('', inplace=True)  # ✅ Replace NaN with empty string

# TF-IDF Vectorizer
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df['text'])

# Save the vectorizer
MODEL_DIR = os.path.join(os.path.dirname(__file__), "../models")
os.makedirs(MODEL_DIR, exist_ok=True)

with open(os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl"), "wb") as f:
    pickle.dump(vectorizer, f)

print("Vectorizer saved successfully!")

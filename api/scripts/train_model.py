import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
import pickle
import os

# Paths
DATA_PATH = "api/dataset/twitter_sentiment.csv"
MODEL_PATH = "api/models/sentiment_model.pkl"
VECTORIZER_PATH = "api/models/tfidf_vectorizer.pkl"

# 🔥 1. Load dataset
print("\n>> Loading dataset...")
df = pd.read_csv(DATA_PATH)

# Clean data
print(f"Dataset columns: {df.columns}")
print(df.head())

# Remove NaN values
print(f"Found {df.isnull().sum().sum()} NaN values. Removing them...")
df.dropna(inplace=True)

# Detect the target column
if 'label' in df.columns:
    target_column = 'label'
else:
    target_column = 'sentiment'
print(f"\n>> Using target column: '{target_column}'")

# Split data
X = df['text']
y = df[target_column]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 🔥 2. Vectorize the text data
print("\n>> Vectorizing text data...")
vectorizer = TfidfVectorizer()

# Fit and transform the training data
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

# 🔥 3. Train the model
print("\n>> Training model...")
model = MultinomialNB()
model.fit(X_train_vectorized, y_train)

# 🔥 4. Evaluate the model
y_pred = model.predict(X_test_vectorized)
accuracy = accuracy_score(y_test, y_pred)

print("\n✅ Model trained successfully!")
print(f"Accuracy: {accuracy:.4f}\n")

# Print classification report
print("Classification Report:")
print(classification_report(y_test, y_pred))

# 🔥 5. Save the model and vectorizer properly
os.makedirs("api/models", exist_ok=True)

# Save the model
with open(MODEL_PATH, "wb") as model_file:
    pickle.dump(model, model_file)
print(f"\n🔥 Model saved successfully at '{MODEL_PATH}'")

# Save the vectorizer
with open(VECTORIZER_PATH, "wb") as vectorizer_file:
    pickle.dump(vectorizer, vectorizer_file)
print(f"🔥 Vectorizer saved successfully at '{VECTORIZER_PATH}'")

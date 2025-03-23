import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv("../dataset/twitter_sentiment.csv")

# Preprocess text
from preprocess import preprocess_text
df["cleaned_text"] = df["text"].apply(preprocess_text)

# Convert labels
label_mapping = {"positive": 1, "negative": 0, "neutral": 2}
df["sentiment"] = df["sentiment"].map(label_mapping)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(df["cleaned_text"], df["sentiment"], test_size=0.2, random_state=42)

# TF-IDF Vectorization
vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Train Logistic Regression Model
model = LogisticRegression()
model.fit(X_train_tfidf, y_train)

# Save Model
with open("../models/sentiment_model.pkl", "wb") as f:
    pickle.dump(model, f)

# Save TF-IDF Vectorizer
with open("../models/tfidf_vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

# Evaluate Model
y_pred = model.predict(X_test_tfidf)
print("Accuracy:", accuracy_score(y_test, y_pred))

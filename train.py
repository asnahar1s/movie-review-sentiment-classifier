"""
train.py — Train and save the Naive Bayes sentiment model using NLTK movie_reviews.
Run this ONCE before starting the app: python train.py
"""

import nltk
import pickle
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report

# ── Download required NLTK data ──────────────────────────────────────────────
print("Downloading NLTK data...")
nltk.download("movie_reviews", quiet=True)
nltk.download("stopwords", quiet=True)
from nltk.corpus import movie_reviews, stopwords

# ── Load dataset ─────────────────────────────────────────────────────────────
print("Loading movie_reviews dataset...")
stop_words = set(stopwords.words("english"))

texts, labels = [], []
for category in movie_reviews.categories():          # 'pos' or 'neg'
    for fileid in movie_reviews.fileids(category):
        words = movie_reviews.words(fileid)
        # Basic cleaning: lowercase, remove stopwords & short tokens
        cleaned = " ".join(
            w.lower() for w in words
            if w.isalpha() and w.lower() not in stop_words and len(w) > 2
        )
        texts.append(cleaned)
        labels.append(1 if category == "pos" else 0)   # 1=positive, 0=negative

print(f"  Loaded {len(texts)} reviews ({labels.count(1)} positive, {labels.count(0)} negative)")

# ── Vectorize with TF-IDF ────────────────────────────────────────────────────
vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
X = vectorizer.fit_transform(texts)

# ── Train / Test split ───────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, labels, test_size=0.2, random_state=42, stratify=labels
)

# ── Train Naive Bayes ────────────────────────────────────────────────────────
print("Training Naive Bayes classifier...")
model = MultinomialNB(alpha=0.5)
model.fit(X_train, y_train)

# ── Evaluate ─────────────────────────────────────────────────────────────────
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
f1  = f1_score(y_test, y_pred)

print("\n── Evaluation Results ──────────────────────────")
print(f"  Accuracy : {acc:.4f}  ({acc*100:.1f}%)")
print(f"  F1 Score : {f1:.4f}")
print("\n  Full Report:")
print(classification_report(y_test, y_pred, target_names=["Negative", "Positive"]))

# ── Save model & vectorizer ───────────────────────────────────────────────────
os.makedirs("model", exist_ok=True)
with open("model/model.pkl", "wb") as f:
    pickle.dump(model, f)
with open("model/vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

print("✓ Model saved to model/model.pkl")
print("✓ Vectorizer saved to model/vectorizer.pkl")
print("\nYou can now run: python app.py")

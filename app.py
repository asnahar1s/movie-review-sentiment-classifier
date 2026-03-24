"""
app.py — Flask backend for the Sentiment Analysis Tool.
Run with: python app.py
"""

import pickle
import re
import os
import nltk
from flask import Flask, render_template, request, jsonify

nltk.download("stopwords", quiet=True)
from nltk.corpus import stopwords

app = Flask(__name__)
stop_words = set(stopwords.words("english"))

# ── Load saved model & vectorizer ────────────────────────────────────────────
MODEL_PATH = os.path.join("model", "model.pkl")
VEC_PATH   = os.path.join("model", "vectorizer.pkl")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        "Model not found. Please run 'python train.py' first."
    )

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)
with open(VEC_PATH, "rb") as f:
    vectorizer = pickle.load(f)

print("✓ Model loaded successfully.")

# ── Text cleaning helper ──────────────────────────────────────────────────────
def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)      # remove URLs
    text = re.sub(r"[^a-z\s]", "", text)             # keep only letters
    tokens = text.split()
    tokens = [t for t in tokens if t not in stop_words and len(t) > 2]
    return " ".join(tokens)

# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    text = data.get("text", "").strip()

    if not text:
        return jsonify({"error": "No text provided"}), 400

    cleaned   = clean_text(text)
    if not cleaned:
        return jsonify({"error": "Text too short or contains no meaningful words."}), 400

    vec       = vectorizer.transform([cleaned])
    label     = model.predict(vec)[0]             # 0 or 1
    proba     = model.predict_proba(vec)[0]       # [neg_prob, pos_prob]

    sentiment = "Positive" if label == 1 else "Negative"
    confidence = float(max(proba)) * 100

    return jsonify({
        "sentiment":  sentiment,
        "confidence": round(confidence, 1),
        "pos_prob":   round(float(proba[1]) * 100, 1),
        "neg_prob":   round(float(proba[0]) * 100, 1),
    })

if __name__ == "__main__":
    print("Starting Sentiment Analysis server at http://127.0.0.1:5000")
    app.run(debug=True)

# 🎬Movie-review-sentiment-classifier
A Python-based sentiment analysis web app that predicts whether a given text is **Positive** or **Negative** using Machine Learning. Built as part of my AI Internship at **SyntecxHub** - Task 3

---

## What It Does
- Loads and preprocesses labeled text data from the NLTK Movie Reviews dataset
- Cleans and tokenizes text (removes stopwords, punctuation, lowercases)
- Converts text to numeric features using **TF-IDF Vectorizer**
- Trains a **Naive Bayes classifier** to predict sentiment
- Provides a beautiful web UI to input any text and instantly see the predicted sentiment with confidence scores

---

## How It Works
Every text input goes through 4 steps:

1. **Preprocessing** — Clean the text, remove noise and stopwords
2. **Vectorization** — Convert text to TF-IDF numeric features
3. **Prediction** — Naive Bayes model classifies as Positive or Negative
4. **Result** — Confidence score shown as a percentage with a visual bar

---

## Project Structure
```
movie-review-sentiment-classifier/
├── train.py              # Trains and saves the ML model
├── app.py                # Flask server that handles predictions
├── requirements.txt      # Required Python libraries
└── templates/
    └── index.html        # Web UI frontend
```

---

## Sample Output
```
"This film was an absolute masterpiece"   → ✅ Positive · 91.3%
"Worst movie I have ever seen"            → ❌ Negative · 87.7%
"The acting was brilliant throughout"     → ✅ Positive · 83.4%
"Terrible plot and a waste of time"       → ❌ Negative · 79.1%
```

---

## 🖼️ Screenshots

<img width="1878" height="903" alt="Screenshot 2026-03-24 150817" src="https://github.com/user-attachments/assets/c266bb60-dbcf-40d2-94df-095ce5633bea" />

<img width="1875" height="908" alt="image" src="https://github.com/user-attachments/assets/72c24da5-6d43-43de-945c-b7e5e427e054" />



---

## Tech Stack
- **Language:** Python 3.x
- **Framework:** Flask
- **ML Library:** scikit-learn
- **Dataset:** NLTK Movie Reviews Corpus
- **Vectorizer:** TF-IDF (Term Frequency–Inverse Document Frequency)
- **Classifier:** Multinomial Naive Bayes
- **Frontend:** HTML, CSS, JavaScript

---

## 👤 Author
**Asna Haris** — AI Intern at SyntecxHub

- GitHub: [asnahar1s](https://github.com/asnahar1s)
- LinkedIn: [asna-haris-684058319](https://www.linkedin.com/in/asna-haris-684058319)

---

## 📄 License
This project is built as part of my AI Internship Program.  
Organization: SyntecxHub  
Website: https://syntecxhub.com/

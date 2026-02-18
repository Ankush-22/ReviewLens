from flask import Flask, render_template, request
import joblib
import re
import nltk
import numpy as np
from scipy.sparse import hstack
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

app = Flask(__name__)

# 1. Load the hybrid model and vectorizer
model = joblib.load('models/fake_review_model.pkl')
tfidf = joblib.load('models/tfidf_vectorizer.pkl')

# Initialize NLP tools
nltk.download('stopwords')
nltk.download('wordnet')
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def clean_review(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text.lower())
    words = [lemmatizer.lemmatize(w) for w in text.split() if w not in stop_words]
    return " ".join(words)

# NEW: Stylometry Extractor for the Web App
def extract_stylometry(text):
    words = text.split()
    if len(words) == 0: return [0, 0, 0]
    diversity = len(set(words)) / len(words)
    punct_count = (text.count('!') + text.count('?')) / len(words)
    pronouns = ['i', 'me', 'my', 'we', 'our']
    pronoun_count = sum(1 for w in words if w.lower() in pronouns) / len(words)
    return [diversity, punct_count, pronoun_count]

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        raw_review = request.form['review']
        
        # 1. Preprocess the text
        cleaned = clean_review(raw_review)
        text_features = tfidf.transform([cleaned])
        
        # 2. Extract Stylometric features
        div, punct, pron = extract_stylometry(raw_review)
        style_features = np.array([[div, punct, pron]])
        
        # 3. Combine them
        final_features = hstack([text_features, style_features])
        
        # 4. GET PREDICTION (The Class: 0 or 1)
        prediction = model.predict(final_features)[0]
        
        # 5. GET PROBABILITY (The Confidence: % chance)
        probabilities = model.predict_proba(final_features)[0]
        deceptive_prob = probabilities[1] * 100 
        
        # 6. Define the Result string
        result = "DECEPTIVE (Fake)" if prediction == 1 else "TRUTHFUL (Real)"
        
        # 7. Send to HTML
        return render_template('index.html', 
                               prediction_text=f'Verdict: {result}', 
                               confidence=f"{deceptive_prob:.1f}%",
                               div_score=f"{div:.2f}",
                               punct_score=f"{punct:.2f}",
                               pron_score=f"{pron:.2f}",
                               original_review=raw_review)

if __name__ == "__main__":
    app.run(debug=True)
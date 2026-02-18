# ReveiwLens 🕵️‍♂️

A hybrid AI-powered web application designed to detect deceptive hotel reviews using a combination of **Semantic NLP (TF-IDF)** and **Stylometric Analysis**.

## 🚀 Overview
Fake reviews are a significant problem in the digital economy. This project uses machine learning to distinguish between truthful and deceptive reviews by analyzing not just *what* is said, but *how* it is written.

### Key Features:
- **Hybrid Detection Engine**: Combines TF-IDF vectorization with handcrafted stylometric features.
- **Stylometric Metrics**:
    - **Vocabulary Diversity**: Measures the richness of the reviewer's vocabulary.
    - **Punctuation Intensity**: Tracks excessive use of exclamation and question marks.
    - **Personal Pronoun Ratio**: Analyzes the self-focus of the reviewer (often higher in fake reviews).
- **Interactive Web Interface**: Built with Flask for real-time analysis.

---

## 🛠️ Tech Stack
- **Backend**: Python, Flask
- **Machine Learning**: Scikit-learn, Joblib
- **NLP**: NLTK, Regex
- **Frontend**: HTML5, CSS3 (Vanilla)
- **Data Handling**: Pandas, NumPy, Scipy

---

## 📂 Project Structure
```text
├── data/               # Datasets used for training
├── models/             # Saved ML models (.pkl)
├── notebooks/          # Jupyter notebooks for EDA and Training
├── templates/          # HTML templates for the Flask app
├── app.py              # Main Flask application
├── requirements.txt    # Project dependencies
└── .gitignore          # Files excluded from version control
```

---

## ⚙️ Installation & Usage

### 1. Clone the repository
```bash
git clone https://github.com/your-username/ReveiwLens.git
cd ReveiwLens
```

### 2. Create a Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Application
```bash
python app.py
```
Open your browser and navigate to `http://127.0.0.1:5000`.

---

## 🧠 How it Works
1. **Preprocessing**: Text is cleaned, lemmatized, and stop-words are removed.
2. **Feature Extraction**:
   - **TF-IDF**: Captures the importance of specific words.
   - **Stylometry**: Calculates mathematical ratios of language use.
3. **Classification**: A pre-trained hybrid model (Random Forest/SVM) predicts the likelihood of the review being fake.

---

## 📄 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

# 🧠 Sentiment Analysis API

A production-grade, end-to-end **Natural Language Processing (NLP)** application that classifies text as **Positive**, **Negative**, or **Neutral** in real time — deployed live on **AWS EC2**.

Built as a Cloud DevOps Final Evaluation Project combining Machine Learning, REST API development, interactive frontend design, and cloud deployment.

🌐 **Live Demo:** [http://3.91.66.61:8501](http://3.91.66.61:8501)  
🔌 **API Base URL:** [http://3.91.66.61:5000](http://3.91.66.61:5000)

---

## 📸 Preview

| Single Prediction | Batch Prediction |
|---|---|
| Type any sentence and get instant sentiment with confidence % | Analyze multiple sentences at once with color-coded results |

---

## 📌 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [ML Model](#ml-model)
- [API Endpoints](#api-endpoints)
- [Installation & Local Setup](#installation--local-setup)
- [AWS EC2 Deployment](#aws-ec2-deployment)
- [Model Performance](#model-performance)
- [Future Scope](#future-scope)

---

## 📖 Overview

Sentiment Analysis is the automated process of identifying the emotional tone behind text. This project builds a complete pipeline:

1. **Train** a machine learning model on labeled text data
2. **Expose** it via a Flask REST API with clean, extensible endpoints
3. **Visualize** predictions through an interactive Streamlit web interface
4. **Deploy** the entire application on AWS EC2 for global accessibility

The system can classify any English text — product reviews, social media comments, customer feedback, or any free-form sentence — into one of three categories with a confidence percentage and full score breakdown.

---

## ✨ Features

- ✅ Real-time sentiment prediction (Positive / Negative / Neutral)
- ✅ Confidence % with full 3-class probability breakdown
- ✅ Single text prediction via REST API
- ✅ Batch prediction — analyze multiple sentences in one API call
- ✅ Interactive Streamlit UI with clean professional design
- ✅ Flask REST API with 4 endpoints returning JSON responses
- ✅ TF-IDF vectorization with unigrams, bigrams, and trigrams
- ✅ Deployed on AWS EC2 — publicly accessible 24/7
- ✅ Version controlled on GitHub with clean commit history

---

## 🛠 Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.x |
| ML Model | Logistic Regression (scikit-learn) |
| Feature Extraction | TF-IDF Vectorizer (ngram 1-3, 15k features) |
| Backend API | Flask |
| Frontend UI | Streamlit |
| Cloud Platform | AWS EC2 (Ubuntu 22.04 LTS, t2.micro) |
| Version Control | Git + GitHub |
| Process Manager | nohup (background process management) |
| Dependencies | scikit-learn, numpy, flask, streamlit, gunicorn |

---

## 📁 Project Structure

```
sentiment-api/
│
├── app.py                  # Flask REST API — 4 prediction endpoints
├── streamlit_app.py        # Streamlit frontend — interactive UI
├── train.py                # ML model training script
├── requirements.txt        # Python package dependencies
├── setup_ec2.sh            # Automated EC2 deployment script
├── DEPLOYMENT_GUIDE.md     # Step-by-step deployment instructions
│
└── model/                  # Generated after running train.py
    ├── sentiment_model.pkl # Trained Logistic Regression model
    └── vectorizer.pkl      # Fitted TF-IDF vectorizer
```

---

## 🤖 ML Model

### Algorithm — Logistic Regression

Logistic Regression is a supervised classification algorithm that learns a decision boundary between classes. For each input, it computes a weighted sum of TF-IDF features and applies the **Softmax function** to produce three probabilities (one per class) that always sum to 100%. The class with the highest probability is the prediction and its probability is the confidence %.

### Feature Engineering — TF-IDF Vectorization

TF-IDF (Term Frequency — Inverse Document Frequency) converts raw text into numerical vectors:

- **TF** — how often a word appears in the sentence
- **IDF** — how rare the word is across all training sentences
- **TF-IDF Score = TF × IDF** — high scores for meaningful, unique words; near-zero for common words like "the", "is", "a"

**Configuration used:**
```python
TfidfVectorizer(
    ngram_range=(1, 3),      # captures single words, pairs, and triplets
    max_features=15000,       # top 15,000 most informative terms
    sublinear_tf=True,        # log-scale TF to reduce impact of very frequent terms
    strip_accents='unicode',  # normalize special characters
)
```

The `ngram_range=(1,3)` is particularly important — it means "not good" is treated as a unit (negative), different from "good" alone (positive).

### Model Configuration

```python
LogisticRegression(
    max_iter=3000,   # training iterations
    C=3.0,           # regularization — higher = trust training data more
    solver='lbfgs'   # optimization algorithm for multiclass problems
)
```

### Training Data

- ~300 labeled sentences (100 per class)
- Balanced dataset — equal representation of all three classes
- 85% training / 15% test split with stratification

---

## 🔌 API Endpoints

Base URL: `http://3.91.66.61:5000`

### GET /
Returns API information and lists all available endpoints.

```json
{
  "message": "Sentiment Analysis API",
  "endpoints": {
    "GET /health": "Health check",
    "POST /predict": "Predict sentiment of text",
    "POST /predict-batch": "Batch prediction"
  }
}
```

---

### GET /health
Health check to confirm the server and model are running.

```json
{
  "status": "healthy",
  "model": "TF-IDF + Logistic Regression"
}
```

---

### POST /predict
Predict sentiment for a single piece of text.

**Request:**
```json
{
  "text": "I absolutely love this product!"
}
```

**Response:**
```json
{
  "text": "I absolutely love this product!",
  "sentiment": "positive",
  "confidence": "94.21%",
  "scores": {
    "positive": 94.21,
    "neutral": 3.64,
    "negative": 2.15
  }
}
```

---

### POST /predict-batch
Predict sentiment for multiple texts in a single request.

**Request:**
```json
{
  "texts": [
    "I love this product!",
    "Delivery was on time.",
    "Terrible quality, very disappointed."
  ]
}
```

**Response:**
```json
{
  "results": [
    { "text": "I love this product!", "sentiment": "positive", "confidence": "91.3%" },
    { "text": "Delivery was on time.", "sentiment": "neutral", "confidence": "83.7%" },
    { "text": "Terrible quality, very disappointed.", "sentiment": "negative", "confidence": "88.2%" }
  ],
  "count": 3
}
```

---

## 💻 Installation & Local Setup

### Prerequisites
- Python 3.8 or higher
- pip or Anaconda

### Steps

**1. Clone the repository**
```bash
git clone https://github.com/Urmeet2k24/sentiment-api.git
cd sentiment-api
```

**2. Install dependencies**
```bash
pip install flask scikit-learn numpy streamlit
```

**3. Train the model**
```bash
python train.py
```
This generates `model/sentiment_model.pkl` and `model/vectorizer.pkl`.

**4. Run the Streamlit UI**
```bash
python -m streamlit run streamlit_app.py
```
Open: [http://localhost:8501](http://localhost:8501)

**5. Run the Flask API (optional)**
```bash
python app.py
```
Open: [http://localhost:5000](http://localhost:5000)

---

## ☁️ AWS EC2 Deployment

### Infrastructure
| Setting | Value |
|---|---|
| Cloud Provider | AWS |
| Service | EC2 (Elastic Compute Cloud) |
| Instance Type | t2.micro (Free Tier) |
| OS | Ubuntu Server 22.04 LTS |
| Public IP | 3.91.66.61 |

### Security Group Rules
| Port | Protocol | Purpose |
|---|---|---|
| 22 | TCP | SSH access |
| 5000 | TCP | Flask REST API |
| 8501 | TCP | Streamlit frontend |

### Deployment Steps

**1. Connect to EC2 via EC2 Instance Connect (browser terminal)**

**2. Install dependencies**
```bash
sudo apt update -y
sudo apt install python3 python3-pip git -y
```

**3. Clone repository**
```bash
git clone https://github.com/Urmeet2k24/sentiment-api.git
cd sentiment-api
```

**4. Install Python packages**
```bash
pip3 install flask scikit-learn numpy streamlit gunicorn --break-system-packages
```

**5. Train the model**
```bash
python3 train.py
```

**6. Start Streamlit (background)**
```bash
nohup python3 -m streamlit run streamlit_app.py --server.port 8501 --server.address 0.0.0.0 > streamlit.log 2>&1 &
```

**7. Start Flask API (background)**
```bash
nohup python3 app.py > flask.log 2>&1 &
```

**8. Access the live application**
- Streamlit UI: `http://3.91.66.61:8501`
- Flask API: `http://3.91.66.61:5000`

---

## 📊 Model Performance

| Metric | Score |
|---|---|
| Test Accuracy | 93.5% |
| Cross-Validation Accuracy | 88.8% (±5.2%) |
| Negative F1-Score | 78% |
| Neutral F1-Score | 90% |
| Positive F1-Score | 70% |
| Training Time | < 1 second |
| Dataset Size | ~300 sentences |

---

## 🔭 Future Scope

- **BERT / Transformers** — Replace Logistic Regression with BERT for significantly higher accuracy and sarcasm/context handling
- **Multilingual Support** — Extend the pipeline to support Hindi, Spanish, and other languages
- **Docker & Kubernetes** — Containerize services for consistent deployment and auto-scaling
- **CI/CD Pipeline** — Automate deployment to EC2 on every GitHub push using GitHub Actions
- **Larger Dataset** — Train on thousands of real-world labeled reviews for better generalization
- **Emotion Detection** — Extend beyond 3 classes to detect specific emotions like anger, joy, fear, and surprise

---

## 👩‍💻 Author

**Urmeet Kaur**  
Enrollment: 2310992234  
B.E. Final Evaluation — Cloud DevOps  

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

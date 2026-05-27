# 🚀 Sentiment Analysis API — AWS EC2 Deployment Guide
## Stack: Python · Flask · scikit-learn · TF-IDF · Logistic Regression

---

## 📁 Project Structure
```
sentiment-api/
├── app.py            ← Flask REST API
├── train.py          ← Model training script
├── requirements.txt  ← Python dependencies
├── setup_ec2.sh      ← EC2 auto-setup script
└── model/            ← Created after training
    ├── sentiment_model.pkl
    └── vectorizer.pkl
```

---

## ⚡ PART 1 — Launch EC2 Instance (5 minutes)

1. Go to **AWS Console → EC2 → Launch Instance**
2. Settings:
   - **Name:** `sentiment-api`
   - **AMI:** Ubuntu Server 22.04 LTS (Free Tier eligible) ✅
   - **Instance type:** `t2.micro` (Free Tier) ✅
   - **Key pair:** Create new → download `.pem` file → SAVE IT!
   - **Security Group — Add these inbound rules:**
     | Type | Port | Source |
     |------|------|--------|
     | SSH  | 22   | My IP  |
     | Custom TCP | 5000 | 0.0.0.0/0 |
3. Click **Launch Instance**
4. Wait ~1 minute for it to say **Running**
5. Copy the **Public IPv4 address** (e.g. `3.91.45.120`)

---

## ⚡ PART 2 — Upload Project Files (3 minutes)

### On your LOCAL machine (Terminal / CMD):

```bash
# Fix key permissions (Mac/Linux only)
chmod 400 your-key.pem

# Upload the entire project folder to EC2
scp -i your-key.pem -r /path/to/sentiment-api ubuntu@YOUR_EC2_IP:~/
```

> **Windows users:** Use WinSCP or Git Bash for the above commands.

---

## ⚡ PART 3 — SSH into EC2 & Deploy (5 minutes)

```bash
# SSH into your instance
ssh -i your-key.pem ubuntu@YOUR_EC2_IP

# Go into the project folder
cd sentiment-api

# Make the setup script executable and run it
chmod +x setup_ec2.sh
./setup_ec2.sh
```

That's it! The script will:
- Install Python and dependencies
- Train the model
- Start the API with Gunicorn

---

## ⚡ PART 4 — Test Your Live API

Replace `YOUR_EC2_IP` with your actual IP.

### Health Check
```bash
curl http://YOUR_EC2_IP:5000/health
```

### Single Prediction
```bash
curl -X POST http://YOUR_EC2_IP:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "I absolutely love this product!"}'
```

Expected response:
```json
{
  "text": "I absolutely love this product!",
  "sentiment": "positive",
  "confidence": "94.21%",
  "scores": {
    "negative": 2.15,
    "neutral": 3.64,
    "positive": 94.21
  }
}
```

### Batch Prediction
```bash
curl -X POST http://YOUR_EC2_IP:5000/predict-batch \
  -H "Content-Type: application/json" \
  -d '{"texts": ["Great product!", "It was okay", "Terrible experience"]}'
```

---

## 🔁 Useful Commands on EC2

```bash
# Check if gunicorn is running
ps aux | grep gunicorn

# Stop gunicorn
pkill gunicorn

# Restart gunicorn manually
source venv/bin/activate
gunicorn --bind 0.0.0.0:5000 --workers 2 --daemon app:app

# View logs
tail -f gunicorn.log
```

---

## 🎯 What to Say in Your Evaluation

**Project:** REST API for NLP-based Sentiment Analysis deployed on AWS EC2

**Tech Stack:**
- **ML Model:** Logistic Regression with TF-IDF vectorization (scikit-learn)
- **API Framework:** Flask with Gunicorn WSGI server
- **Cloud:** AWS EC2 (Ubuntu 22.04, t2.micro)
- **Features:** Single prediction, batch prediction, confidence scores

**ML Concepts used:**
- TF-IDF (Term Frequency-Inverse Document Frequency) for feature extraction
- N-gram tokenization (unigrams + bigrams)
- Multinomial Logistic Regression for 3-class classification
- Train/test split with stratification

---

## ❗ Troubleshooting

| Problem | Fix |
|---------|-----|
| Port 5000 not accessible | Check EC2 Security Group has port 5000 open |
| SSH permission denied | Run `chmod 400 your-key.pem` first |
| Module not found | Run `source venv/bin/activate` before anything |
| App not running | Run `ps aux | grep gunicorn` to check |

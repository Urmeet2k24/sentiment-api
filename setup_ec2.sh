#!/bin/bash
# ============================================================
#  EC2 Setup Script — Sentiment Analysis API
#  Run this on your EC2 instance after SSH-ing in
# ============================================================

set -e  # stop on any error

echo "===================================="
echo " Step 1: Update system packages"
echo "===================================="
sudo apt-get update -y
sudo apt-get install -y python3 python3-pip python3-venv git unzip

echo "===================================="
echo " Step 2: Copy project files"
echo "===================================="
# If you uploaded a zip, unzip it:
# unzip sentiment-api.zip
# cd sentiment-api

echo "===================================="
echo " Step 3: Create virtual environment"
echo "===================================="
python3 -m venv venv
source venv/bin/activate

echo "===================================="
echo " Step 4: Install dependencies"
echo "===================================="
pip install --upgrade pip
pip install -r requirements.txt

echo "===================================="
echo " Step 5: Train the model"
echo "===================================="
python3 train.py

echo "===================================="
echo " Step 6: Test the app locally"
echo "===================================="
echo "Starting app briefly to verify..."
timeout 5 python3 app.py &
sleep 3
curl -s http://localhost:5000/health || echo "App started OK"

echo "===================================="
echo " Step 7: Run with Gunicorn (production)"
echo "===================================="
echo "Starting Gunicorn on port 5000..."
gunicorn --bind 0.0.0.0:5000 --workers 2 --daemon app:app

echo ""
echo "=============================================="
echo " DEPLOYMENT COMPLETE!"
echo " Your API is live at: http://$(curl -s http://169.254.169.254/latest/meta-data/public-ipv4):5000"
echo "=============================================="

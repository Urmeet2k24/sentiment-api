from flask import Flask, request, jsonify
import pickle
import os

app = Flask(__name__)

# Load model and vectorizer at startup
MODEL_PATH = "model/sentiment_model.pkl"
VECTORIZER_PATH = "model/vectorizer.pkl"

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

with open(VECTORIZER_PATH, "rb") as f:
    vectorizer = pickle.load(f)

LABELS = {0: "negative", 1: "neutral", 2: "positive"}

@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "Sentiment Analysis API",
        "endpoints": {
            "POST /predict": "Predict sentiment of text",
            "GET /health": "Health check"
        }
    })

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "healthy", "model": "TF-IDF + Logistic Regression"})

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    if not data or "text" not in data:
        return jsonify({"error": "Please provide 'text' field in JSON body"}), 400

    text = data["text"]
    if not text.strip():
        return jsonify({"error": "Text cannot be empty"}), 400

    # Vectorize and predict
    X = vectorizer.transform([text])
    pred = model.predict(X)[0]
    proba = model.predict_proba(X)[0]

    confidence = round(float(max(proba)) * 100, 2)
    sentiment = LABELS[pred]

    return jsonify({
        "text": text,
        "sentiment": sentiment,
        "confidence": f"{confidence}%",
        "scores": {
            "negative": round(float(proba[0]) * 100, 2),
            "neutral":  round(float(proba[1]) * 100, 2),
            "positive": round(float(proba[2]) * 100, 2)
        }
    })

@app.route("/predict-batch", methods=["POST"])
def predict_batch():
    data = request.get_json()

    if not data or "texts" not in data:
        return jsonify({"error": "Please provide 'texts' array in JSON body"}), 400

    texts = data["texts"]
    if not isinstance(texts, list) or len(texts) == 0:
        return jsonify({"error": "'texts' must be a non-empty list"}), 400

    X = vectorizer.transform(texts)
    preds = model.predict(X)
    probas = model.predict_proba(X)

    results = []
    for i, text in enumerate(texts):
        proba = probas[i]
        results.append({
            "text": text,
            "sentiment": LABELS[preds[i]],
            "confidence": f"{round(float(max(proba)) * 100, 2)}%"
        })

    return jsonify({"results": results, "count": len(results)})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)

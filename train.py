"""
Train a Sentiment Analysis model using TF-IDF + Logistic Regression.
Saves model and vectorizer to the model/ directory.
"""

import os
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

os.makedirs("model", exist_ok=True)

# Label: 0=negative, 1=neutral, 2=positive
positives = [
    "I absolutely love this product, it works great!",
    "This is the best experience I've ever had.",
    "Amazing quality, totally worth the price.",
    "Fantastic service, will definitely come back!",
    "I'm so happy with my purchase, highly recommend.",
    "The movie was wonderful and very entertaining.",
    "Great customer support, resolved my issue instantly.",
    "Excellent work, exceeded all my expectations.",
    "This app is incredibly useful and easy to use.",
    "I had a brilliant time, everything was perfect.",
    "Very satisfied with the outcome, kudos to the team.",
    "Outstanding performance, truly impressive results.",
    "The food was delicious and the staff was very friendly.",
    "Super fast delivery, product is exactly as described.",
    "Five stars! I couldn't be happier with this.",
    "Loved every bit of it, will recommend to friends.",
    "The quality is top-notch, very well made.",
    "Best decision I ever made, so glad I bought this.",
    "Incredible value for money, works perfectly.",
    "I am thoroughly impressed, will buy again!",
    "This exceeded every expectation I had going in.",
    "Phenomenal results, I am blown away.",
    "Delightful experience from start to finish.",
    "So well designed, a pleasure to use every day.",
    "Superb craftsmanship and attention to detail.",
    "Really happy I found this, it changed my routine.",
    "Works like a charm, no issues whatsoever.",
    "Hands down the best purchase I made this year.",
    "Brilliant, fast, and reliable — love it.",
    "Could not be more pleased with the result.",
    "Top quality product, delivered on time, perfect.",
    "The performance is exceptional, highly recommend.",
    "Everything went smoothly, absolutely great!",
    "Such a pleasant surprise, way better than expected.",
    "Very efficient and easy to set up, works flawlessly.",
]

neutrals = [
    "The product arrived on time.",
    "It works as described in the manual.",
    "The package was delivered yesterday.",
    "I received the item in standard condition.",
    "The service was okay, nothing special.",
    "The meeting was scheduled for 3pm.",
    "The product is available in three colors.",
    "I used the app to check my balance.",
    "The instructions were included in the box.",
    "The store opens at 9am every weekday.",
    "I ordered the medium size.",
    "The report was submitted on Monday.",
    "The update is available for download.",
    "Customer support responded within 24 hours.",
    "The item weighs approximately 2 kilograms.",
    "The software version is 3.1.2.",
    "The event starts at 6pm.",
    "The product comes with a one-year warranty.",
    "I followed the steps mentioned in the guide.",
    "The subscription renews every month.",
    "The office is located on the third floor.",
    "The form requires a valid email address.",
    "The system will restart after the update.",
    "The shipment is currently in transit.",
    "The file size is approximately 50 megabytes.",
    "I called the helpline and left a message.",
    "The battery lasts about eight hours on average.",
    "The confirmation email was sent to my inbox.",
    "The course is six weeks long.",
    "The model number is printed on the back.",
    "I registered an account on the website.",
    "The transaction was processed successfully.",
    "The nearest branch is 3 kilometers away.",
    "I set the alarm for 7am.",
    "The meeting agenda was shared in advance.",
]

negatives = [
    "This is the worst product I've ever bought.",
    "Terrible service, I am very disappointed.",
    "Total waste of money, completely useless.",
    "I hate this so much, it broke after one day.",
    "Very poor quality, would not recommend at all.",
    "Awful experience, the staff was rude.",
    "The delivery was extremely late and damaged.",
    "I want a refund, this is unacceptable.",
    "Horrible customer service, never again.",
    "The product stopped working within a week.",
    "Extremely frustrating, nothing works as promised.",
    "I am very unhappy with this purchase.",
    "Complete garbage, a total scam.",
    "Worst app ever, crashes constantly.",
    "Disappointed beyond words, very bad quality.",
    "Never buying from this company again.",
    "The instructions were useless and confusing.",
    "Defective item, customer support ignored me.",
    "I regret buying this, it's a ripoff.",
    "Pathetic product, broke on first use.",
    "Absolutely terrible, do not waste your money.",
    "The worst customer experience of my life.",
    "Faulty from day one, nobody helped me.",
    "Overpriced and underdelivered, very disappointed.",
    "This product is a complete disaster.",
    "Rude staff, zero empathy, avoid at all costs.",
    "Broken on arrival, getting a refund immediately.",
    "Nothing but problems since I bought this.",
    "I am furious, this is false advertising.",
    "Garbage quality, fell apart after two uses.",
    "Support team is useless and unhelpful.",
    "Failed to work as advertised, very annoying.",
    "Disgusting quality, not worth a single penny.",
    "Frustrating to set up and it still doesn't work.",
    "Worst experience ever, completely let down.",
]

texts  = positives + neutrals + negatives
labels = [2]*len(positives) + [1]*len(neutrals) + [0]*len(negatives)

X_train, X_test, y_train, y_test = train_test_split(
    texts, labels, test_size=0.2, random_state=42, stratify=labels
)

vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=5000, sublinear_tf=True)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec  = vectorizer.transform(X_test)

model = LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs")
model.fit(X_train_vec, y_train)

y_pred = model.predict(X_test_vec)
acc = accuracy_score(y_test, y_pred)

print("=" * 50)
print(f"  Model Training Complete")
print(f"  Test Accuracy: {acc * 100:.1f}%")
print("=" * 50)
print(classification_report(y_test, y_pred, target_names=["negative", "neutral", "positive"]))

with open("model/sentiment_model.pkl", "wb") as f:
    pickle.dump(model, f)
with open("model/vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

print("Model saved to model/sentiment_model.pkl")
print("Vectorizer saved to model/vectorizer.pkl")

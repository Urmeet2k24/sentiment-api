import os, pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

os.makedirs("model", exist_ok=True)

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
    "This product is amazing and I love it so much.",
    "What an amazing product, totally worth it!",
    "I enjoy using this every single day, it's wonderful.",
    "Wow, this is fantastic, best thing I bought!",
    "So good, I recommended it to all my friends.",
    "Incredible product, very happy with my decision.",
    "Great value, great quality, great experience overall.",
    "This is awesome, exceeded all my hopes.",
    "I feel great about this purchase, very positive experience.",
    "Loved it from day one, works perfectly every time.",
    "Superb product, does exactly what it promises.",
    "I'm thrilled with this, absolutely recommend it.",
    "Best product in its category, no doubt about it.",
    "Very impressed, it works beautifully and efficiently.",
    "Happy customer here, will definitely order again!",
    "This thing is brilliant, makes my life so much easier.",
    "Amazing experience, staff was kind and very helpful.",
    "Product is great, arrived quickly, very satisfied.",
    "So pleased with this purchase, it's exactly what I wanted.",
    "Everything about this is positive, highly recommend!",
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
    "I read the terms and conditions before signing.",
    "The package includes two items.",
    "The receipt was emailed after purchase.",
    "I downloaded the app from the store.",
    "The temperature today is 22 degrees.",
    "The document was saved in PDF format.",
    "I submitted the form online.",
    "The class meets twice a week.",
    "The item is currently out of stock.",
    "The device supports wireless charging.",
    "I changed my password last week.",
    "The project deadline is next Friday.",
    "The manual is available in English and Hindi.",
    "I checked my order status on the website.",
    "The product has a 30 day return policy.",
    "The account was created successfully.",
    "I updated the app to the latest version.",
    "The file was uploaded to the server.",
    "I picked up the parcel from the post office.",
    "The bill was paid through net banking.",
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
    "Frustrating to set up and it still does not work.",
    "Worst experience ever, completely let down.",
    "This is such a bad product, I hate it.",
    "Do not buy this, it is a complete waste.",
    "Very unhappy, the product is broken and cheap.",
    "Extremely poor quality, very dissatisfied.",
    "I dislike everything about this product.",
    "It failed immediately, very bad experience.",
    "The worst thing I have ever purchased, avoid it.",
    "Zero stars, awful product, rude support team.",
    "Such a disappointment, expected much better.",
    "The product is defective and the company doesn't care.",
    "Not worth it at all, completely regret buying.",
    "Bad quality, bad service, bad experience overall.",
    "I am so frustrated, this never worked properly.",
    "Shocking quality for this price, total rip off.",
    "Terrible in every way, would give zero stars.",
    "Highly disappointed, this product is a failure.",
    "Broken after one use, absolute junk.",
    "The service was atrocious and unhelpful.",
    "Waste of time and money, deeply unsatisfied.",
]

texts  = positives + neutrals + negatives
labels = [2]*len(positives) + [1]*len(neutrals) + [0]*len(negatives)

X_train, X_test, y_train, y_test = train_test_split(
    texts, labels, test_size=0.2, random_state=42, stratify=labels
)

vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=8000, sublinear_tf=True)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec  = vectorizer.transform(X_test)

model = LogisticRegression(max_iter=2000, C=2.0, solver="lbfgs")
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

print("Done! Model saved.")

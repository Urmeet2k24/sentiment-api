import os, pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

os.makedirs("model", exist_ok=True)

positives = [
    "I absolutely love this product",
    "I loved this product so much",
    "i loved this product",
    "I love this",
    "this is amazing",
    "I really loved it",
    "loved every bit of it",
    "I enjoy using this every day",
    "this product is amazing",
    "what an amazing product",
    "this is the best product ever",
    "I am so happy with this purchase",
    "highly recommend this to everyone",
    "fantastic quality and great service",
    "exceeded all my expectations",
    "absolutely wonderful experience",
    "great value for money",
    "works perfectly, very satisfied",
    "brilliant product, will buy again",
    "outstanding performance, love it",
    "five stars, couldn't be happier",
    "super fast delivery, great product",
    "top notch quality, very impressive",
    "best purchase I made this year",
    "incredible, works like a charm",
    "so glad I bought this, amazing",
    "very happy with this, great stuff",
    "excellent product, no complaints",
    "I'm thrilled with this purchase",
    "delightful experience, highly recommend",
    "phenomenal results, blown away",
    "superb craftsmanship, very well made",
    "this changed my life for the better",
    "so well designed, pleasure to use",
    "I feel great about this purchase",
    "really happy I found this product",
    "love the quality, love the service",
    "wow this is fantastic, best thing ever",
    "so good, recommended to all my friends",
    "very efficient and works flawlessly",
    "perfect in every way, thank you",
    "the best decision I ever made buying this",
    "I am thoroughly impressed with this",
    "great product great price great service",
    "awesome awesome awesome, love it",
    "could not be more pleased",
    "this product rocks, absolutely love it",
    "everything is perfect, very happy",
    "I am a fan, will definitely order again",
    "this is wonderful, works beyond expectations",
    "such a pleasant surprise, way better than expected",
    "hands down the best product in its category",
    "I enjoy this every single day without any issues",
    "very pleased, does exactly what it promises",
    "absolutely recommend, great in every way",
    "it works beautifully, super impressed",
    "love love love this product",
    "this made me very happy",
    "positive experience from start to finish",
    "great item, fast delivery, happy customer",
    "I liked this product a lot",
    "pretty good product overall",
    "I like it, works well for me",
    "good quality, happy with purchase",
    "nice product, would buy again",
    "decent quality and good price",
    "satisfied with this purchase",
    "works as expected, good product",
    "quite good, no major issues",
]

neutrals = [
    "the product arrived on time",
    "it works as described in the manual",
    "the package was delivered yesterday",
    "I received the item in standard condition",
    "the service was okay nothing special",
    "the product is available in three colors",
    "I used the app to check my balance",
    "the instructions were included in the box",
    "the store opens at 9am every weekday",
    "I ordered the medium size",
    "the report was submitted on Monday",
    "the update is available for download",
    "customer support responded within 24 hours",
    "the item weighs approximately 2 kilograms",
    "the software version is 3.1.2",
    "the event starts at 6pm",
    "the product comes with a one year warranty",
    "I followed the steps mentioned in the guide",
    "the subscription renews every month",
    "the office is located on the third floor",
    "the form requires a valid email address",
    "the system will restart after the update",
    "the shipment is currently in transit",
    "I called the helpline and left a message",
    "the battery lasts about eight hours on average",
    "the confirmation email was sent to my inbox",
    "the course is six weeks long",
    "the model number is printed on the back",
    "I registered an account on the website",
    "the transaction was processed successfully",
    "I set the alarm for 7am",
    "the meeting agenda was shared in advance",
    "the document was saved in PDF format",
    "I submitted the form online",
    "the class meets twice a week",
    "the item is currently out of stock",
    "the device supports wireless charging",
    "I changed my password last week",
    "the project deadline is next Friday",
    "I checked my order status on the website",
    "the product has a 30 day return policy",
    "the account was created successfully",
    "I updated the app to the latest version",
    "the file was uploaded to the server",
    "the bill was paid through net banking",
    "I picked up the parcel from the post office",
    "the package includes two items",
    "the receipt was emailed after purchase",
    "I downloaded the app from the store",
    "the manual is available in English and Hindi",
    "the item is compatible with most devices",
    "the delivery usually takes three to five days",
    "I contacted support and they replied",
    "the product dimensions are listed on the page",
    "I added the item to my cart",
    "the return process was straightforward",
    "the tracking number was sent via email",
    "I reviewed the terms before purchasing",
    "the product is manufactured in Germany",
    "I compared a few options before deciding",
]

negatives = [
    "this is the worst product I have ever bought",
    "terrible service I am very disappointed",
    "total waste of money completely useless",
    "I hate this so much it broke after one day",
    "very poor quality would not recommend at all",
    "awful experience the staff was rude",
    "the delivery was extremely late and damaged",
    "I want a refund this is unacceptable",
    "horrible customer service never again",
    "the product stopped working within a week",
    "extremely frustrating nothing works as promised",
    "I am very unhappy with this purchase",
    "complete garbage a total scam",
    "worst app ever crashes constantly",
    "disappointed beyond words very bad quality",
    "never buying from this company again",
    "defective item customer support ignored me",
    "I regret buying this it is a ripoff",
    "pathetic product broke on first use",
    "absolutely terrible do not waste your money",
    "the worst customer experience of my life",
    "faulty from day one nobody helped me",
    "overpriced and underdelivered very disappointed",
    "this product is a complete disaster",
    "rude staff zero empathy avoid at all costs",
    "broken on arrival getting a refund immediately",
    "nothing but problems since I bought this",
    "I am furious this is false advertising",
    "garbage quality fell apart after two uses",
    "support team is useless and unhelpful",
    "failed to work as advertised very annoying",
    "not worth a single penny disgusting quality",
    "frustrating to set up and it still does not work",
    "worst experience ever completely let down",
    "this is such a bad product I hate it",
    "do not buy this it is a complete waste",
    "very unhappy the product is broken and cheap",
    "extremely poor quality very dissatisfied",
    "I dislike everything about this product",
    "it failed immediately very bad experience",
    "the worst thing I have ever purchased avoid it",
    "zero stars awful product rude support team",
    "such a disappointment expected much better",
    "bad quality bad service bad experience overall",
    "I am so frustrated this never worked properly",
    "shocking quality for this price total rip off",
    "terrible in every way would give zero stars",
    "highly disappointed this product is a failure",
    "broken after one use absolute junk",
    "the service was atrocious and unhelpful",
    "waste of time and money deeply unsatisfied",
    "I would not recommend this to anyone at all",
    "complete failure of a product avoid",
    "this is rubbish do not buy it",
    "money wasted on this awful product",
    "I am really unhappy with this item",
    "terrible terrible terrible product",
    "does not work at all very bad",
    "so disappointed wish I never bought this",
    "awful quality broke immediately useless",
]

texts  = positives + neutrals + negatives
labels = [2]*len(positives) + [1]*len(neutrals) + [0]*len(negatives)

X_train, X_test, y_train, y_test = train_test_split(
    texts, labels, test_size=0.15, random_state=42, stratify=labels
)

vectorizer = TfidfVectorizer(ngram_range=(1, 3), max_features=10000, sublinear_tf=True, min_df=1)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec  = vectorizer.transform(X_test)

model = LogisticRegression(max_iter=3000, C=5.0, solver="lbfgs")
model.fit(X_train_vec, y_train)

y_pred = model.predict(X_test_vec)
acc = accuracy_score(y_test, y_pred)

print("=" * 50)
print(f"  Test Accuracy: {acc * 100:.1f}%")
print("=" * 50)
print(classification_report(y_test, y_pred, target_names=["negative", "neutral", "positive"]))

# Quick sanity check
test_cases = [
    ("i loved this product", 2),
    ("this is amazing", 2),
    ("terrible waste of money", 0),
    ("the package arrived on time", 1),
    ("I hate this product", 0),
    ("great quality very happy", 2),
]
print("\nSanity checks:")
for text, expected in test_cases:
    X = vectorizer.transform([text])
    pred = model.predict(X)[0]
    status = "✓" if pred == expected else "✗"
    labels_map = {0:"negative", 1:"neutral", 2:"positive"}
    print(f"  {status} '{text}' → {labels_map[pred]} (expected {labels_map[expected]})")

with open("model/sentiment_model.pkl", "wb") as f:
    pickle.dump(model, f)
with open("model/vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

print("\nModel saved successfully.")

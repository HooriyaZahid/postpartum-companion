import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import joblib
import re

# Load data
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

EMOTION_MAP = {
    0: "positive", 1: "positive", 2: "overwhelmed", 3: "overwhelmed",
    4: "positive", 5: "positive", 6: "anxious", 7: "neutral",
    8: "positive", 9: "sad", 10: "overwhelmed", 11: "overwhelmed",
    12: "anxious", 13: "positive", 14: "anxious", 15: "positive",
    16: "sad", 17: "positive", 18: "positive", 19: "anxious",
    20: "positive", 21: "positive", 22: "neutral", 23: "positive",
    24: "sad", 25: "sad", 26: "neutral", 27: "neutral",
}

def map_emotion(label_val):
    nums = re.findall(r'\d+', str(label_val))
    if nums:
        return EMOTION_MAP.get(int(nums[0]), "neutral")
    return "neutral"

train["emotion"] = train["labels"].apply(map_emotion)
test["emotion"] = test["labels"].apply(map_emotion)

print("Emotion distribution:")
print(train["emotion"].value_counts())

vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1,2))
X_train = vectorizer.fit_transform(train["text"])
X_test = vectorizer.transform(test["text"])

model = LogisticRegression(max_iter=1000, class_weight="balanced")
model.fit(X_train, train["emotion"])

preds = model.predict(X_test)
print("\nClassification Report:")
print(classification_report(test["emotion"], preds))

joblib.dump(model, "emotion_model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")
print("\nModel saved!")

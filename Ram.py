# app.py

import streamlit as st
import re
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

nltk.download('stopwords')

# Load model & vectorizer
model = pickle.load(open("sentiment_model.pkl", "rb"))
vectorizer = pickle.load(open("tfidf_vectorizer.pkl", "rb"))

# Text preprocessing
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

def preprocess(text):
    text = text.lower()
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    tokens = text.split()
    tokens = [stemmer.stem(word) for word in tokens if word not in stop_words]
    return " ".join(tokens)

# Streamlit UI
st.title("Sentiment Analysis App")
st.write("Enter a movie review to predict its sentiment")

review = st.text_area("Type your movie review here:")

if st.button("Predict Sentiment"):
    if review.strip() == "":
        st.warning("Please enter a review.")
    else:
        processed = preprocess(review)
        vec = vectorizer.transform([processed])
        pred = model.predict(vec)[0]
        result = "Positive" if pred == 1 else "Negative"
        st.success(f"Predicted Sentiment: *{result}*")

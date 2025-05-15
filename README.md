# Sentiment-Analysis-Streamlit-Project
# Sentiment Analysis Streamlit App
This is a simple Sentiment Analysis web app built using Streamlit and Machine Learning (TF-IDF + Logistic Regression). It allows users to enter movie reviews and get sentiment predictions (Positive or Negative).

## Features

- Clean and interactive Streamlit UI
- Input text box for review entry
- ML model trained using TF-IDF and Logistic Regression
- Real-time Sentiment Prediction
- Ready for deployment on platforms like Streamlit Cloud or Render

## How it Works

1. The model is trained on a dataset of movie reviews labeled as positive or negative.
2. Text is vectorized using TF-IDF (Term Frequency–Inverse Document Frequency).
3. A Logistic Regression model is trained on these vectors.
4. The model is saved and loaded in the Streamlit app for real-time prediction.

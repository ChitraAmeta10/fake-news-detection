import streamlit as st
import joblib

# Load model + vectorizer
model = joblib.load("fake_news_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

st.title("📰 Fake News Detection")

st.write("Enter a news headline and click Predict.")

# Text Input
user_input = st.text_input("News text:")

# Predict Button
if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        x = vectorizer.transform([user_input])
        pred = model.predict(x)[0]   # "fake" or "real"

        if pred == "fake":
            st.error("Fake News")
        else:
            st.success("Real News")

import streamlit as st
import tensorflow as tf
import pickle
from pathlib import Path

#get the Folder containing this python file
BASE_DIR=Path(__file__).resolve().parent

st.title("IMDb movie review Sentiment Analyzer")
st.write("Enter a movie review to predict it's sentiment.")

MODEL_PATH=BASE_DIR/"imdb_sentiment_model.keras"
VECTORIZER_PATH=BASE_DIR/"imdb_vectorizer (1).pkl"

#Load the trained model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

# Load the text_vectorizer
@st.cache_resource
def load_vectorizer():
    with open(VECTORIZER_PATH,"rb")as file:
        return pickle.load(file)

# check whether required files exist
if not MODEL_PATH.exists():
    st.error("Model file not found:imdb_sentiment_model.keras")
elif not VECTORIZER_PATH.exists():
    st.error("Vectorizer file not found:imdb_vectorizer (1).pkl")
else:
    try:
        model=load_model()
        vectorizer=load_vectorizer()
        review=st.text_area("Enter your movie review:")

        if st.button("Analyze Sentiment"):
            if review.strip():
                review_vector=vectorizer(tf.constant([review]))

                prediction=model.predict(review_vector,verbose=0)
                score=float(prediction[0][0])
                if score>=0.5:
                    st.success("Positive review")
                else:
                    st.error("Negative review")

            else:
                st.warning("please enter a movie review.")

    except Exception as e:
        st.error("Something went wrong while loading or predict.")
        st.exception(e)

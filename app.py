import streamlit as st
import joblib
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

# Load the model and tokenizer
try:
    model = load_model('sentiment_analysis_model.h5')
except Exception as e:
    st.error(f"Error loading model: {e}")

try:
    with open('tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
except Exception as e:
    st.error(f"Error loading tokenizer: {e}")


def predict_sentiment(text):
 
    text_sequence = tokenizer.texts_to_sequences([text])
    text_sequence = pad_sequences(text_sequence, maxlen=100)

    
    predicted_rating = model.predict(text_sequence)[0]
    if np.argmax(predicted_rating) == 0:
        return 'Negative'
    elif np.argmax(predicted_rating) == 1:
        return 'Neutral'
    else:
        return 'Positive'

st.title('Sentiment Analysis')
comment = st.text_area('Enter your comment:')
if st.button('Analyze Sentiment'):
    if 'model' in globals() and 'tokenizer' in globals():
        sentiment = predict_sentiment(comment)
        st.write(f'Sentiment: {sentiment}')
    else:
        st.error("Model or tokenizer not loaded properly.")
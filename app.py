import streamlit as st
import joblib
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

# Load the model and tokenizer
model = load_model('sentiment_analysis_model.h5')
tokenizer = joblib.load('tokenizer.pickle')

def predict_sentiment(comment):
    # Tokenize and pad the input text
    text_sequence = tokenizer.texts_to_sequences([comment])
    text_sequence = pad_sequences(text_sequence, maxlen=100)
    # Make a prediction using the trained model
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
    sentiment = predict_sentiment(comment)
    st.write(f'Sentiment: {sentiment}')

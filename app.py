from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model

# Load your sentiment analysis model and tokenizer
model = load_model('sentiment_analysis_model.keras')
tokenizer = joblib.load('tokenizer.pickle')

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

# Define a function to predict the sentiment of input text
def predict_sentiment(text):
    # Tokenize and pad the input text
    text_sequence = tokenizer.texts_to_sequences([text])
    text_sequence = pad_sequences(text_sequence, maxlen=100)

    # Make a prediction using the trained model
    predicted_rating = model.predict(text_sequence)[0]
    if np.argmax(predicted_rating) == 0:
        return 'Negative'
    elif np.argmax(predicted_rating) == 1:
        return 'Neutral'
    else:
        return 'Positive'

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    comment = data['comment']
    sentiment = predict_sentiment(comment)
    return jsonify({'sentiment': sentiment})

if __name__ == '__main__':
    app.run(debug=False)

# Import necessary libraries
from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
import numpy as np

app = Flask(__name__)

# Load the trained model
model_path = 'hate_speech.h5'
model = load_model(model_path)

# Load the local tokenizer
tokenizer_path = 'tokenizer.pickle'  # Adjust the path to your tokenizer file
with open(tokenizer_path, 'rb') as handle:
    tokenizer = pickle.load(handle)

# Define the maximum sequence length
max_seq_length = 128  # Adjust based on your model's requirements

# Define the API endpoint
@app.route('/detect_hate_speech', methods=['POST'])
def detect_hate_speech():
    user_input = request.form['user_input']

    # Tokenize and preprocess user input
    sequences = tokenizer.texts_to_sequences([user_input])
    padded_sequences = pad_sequences(sequences, maxlen=max_seq_length)

    # Make predictions
    prediction = model.predict(padded_sequences)

    # Convert predicted probability to a class label
    predicted_class = "Hate Speech" if prediction[0, 0] >= 0.5 else "Non-Hate Speech"

    # Return the prediction as JSON with CORS headers to allow cross-origin requests
    response = jsonify({'prediction': predicted_class})
    response.headers.add('Access-Control-Allow-Origin', '*')  # Adjust as needed for security in production
    return response

if __name__ == '__main__':
    app.run(debug=True)

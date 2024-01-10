# # Import necessary libraries
# from flask import Flask, render_template, request, jsonify
# from transformers import BertModel, BertTokenizer, BertForSequenceClassification, BertConfig
# import torch

# app = Flask(__name__)

# # Load the trained model
# model_path = 'fine_tuned_model.pt'  
# config_path = 'Fine Tuned Model/config.json'  
# model_config = BertConfig.from_json_file(config_path)

# model = BertForSequenceClassification(config=model_config)
# model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))

# # Load the local tokenizer
# tokenizer_path = 'Fine Tuned Tokenizer/vocab.txt'  
# tokenizer = BertTokenizer(vocab_file=tokenizer_path)

# # # Load the Sagar Sarker Bangla BERT model
# # sagar_sarker_model = BertModel.from_pretrained('sagorsarker/bangla-bert-base')

# # # Create a new instance of BertForSequenceClassification using the Sagar Sarker model's configuration
# # model_config = sagar_sarker_model.config
# # sagar_sarker_classification_model = BertForSequenceClassification(config=model_config)

# # # Specify the path to the local tokenizer
# # tokenizer_path = 'Fine Tuned Tokenizer/vocab.txt'  
# # local_tokenizer = BertTokenizer(vocab_file=tokenizer_path)

# # # Set the Sagar Sarker model's weights to the new classification model
# # sagar_sarker_classification_model.load_state_dict(sagar_sarker_model.state_dict())


# # Define the maximum sequence length
# max_seq_length = 128  # You can adjust this value based on your model's requirements

# # Define the API endpoint
# @app.route('/detect_hate_speech', methods=['POST'])
# def detect_hate_speech():
#     user_input = request.form['user_input']

#     # Tokenize and preprocess user input
#     inputs = tokenizer(user_input, return_tensors="pt", truncation=True, padding=True, max_length=max_seq_length)

#     # Move tensors to the appropriate device
#     input_ids = inputs['input_ids']
#     attention_mask = inputs['attention_mask']

#     # Set the model to evaluation mode
#     model.eval()

#     with torch.no_grad():
#         # Forward pass
#         outputs = model(input_ids, attention_mask=attention_mask)
#         logits = outputs.logits

#     # Get predicted class
#     _, predicted = torch.max(logits, dim=1)

#     # Convert predicted class to a human-readable label
#     predicted_class = "Hate Speech" if predicted.item() == 1 else "Non-Hate Speech"

#     # Return the prediction as JSON with CORS headers to allow cross-origin requests
#     response = jsonify({'prediction': predicted_class})
#     response.headers.add('Access-Control-Allow-Origin', '*')  # Adjust as needed for security in production
#     return response

# if __name__ == '__main__':
#     app.run(debug=True)

# Import necessary libraries
from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
import numpy as np
import pickle


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

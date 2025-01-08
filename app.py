

import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import numpy as np
import pickle
import json
import random
import tensorflow as tf
from tensorflow.keras.models import load_model
from flask import Flask, render_template, request
from textblob import TextBlob
import spacy

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

# Initialize lemmatizer and stopwords
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Load required files
model = load_model('model.h5')
intents = json.loads(open('data.json').read())
words = pickle.load(open('texts.pkl', 'rb'))
classes = pickle.load(open('labels.pkl', 'rb'))

# Load spaCy model for NER (Optional)
nlp = spacy.load("en_core_web_sm")

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words if word not in stop_words]
    return sentence_words

def bow(sentence, words, show_details=True):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
                if show_details:
                    print(f"found in bag: {w}")
    return np.array(bag)

def predict_class(sentence, model):
    p = bow(sentence, words, show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25  # Confidence threshold
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    
    # If no intent is predicted with sufficient confidence, return 'fallback'
    if not results:
        return [{"intent": "fallback", "probability": "0.0"}]
    
    return [{"intent": classes[r[0]], "probability": str(r[1])} for r in results]

def get_response(user_input, intents_json, context):
    # ... (existing code for intent recognition and response selection)

    # Check for looping behavior and implement a timeout
    if response == "I'm here to help. Can you share more details?":
        if context["loop_count"] > 2:
            response = "I'm not quite understanding. Let's try a different approach. What else can I help you with?"
            context["loop_count"] = 0
        else:
            context["loop_count"] += 1

    # Implement a diversity strategy
    if response in recent_responses:
        response = get_alternative_response(intents_json, tag)  # Function to get an alternative response

    return response, context

def chatbot_response(msg):
    ints = predict_class(msg, model)
    response = getResponse(ints, intents)
    return response

def sentiment_analysis(msg):
    analysis = TextBlob(msg)
    if analysis.sentiment.polarity > 0:
        return "Positive"
    elif analysis.sentiment.polarity < 0:
        return "Negative"
    else:
        return "Neutral"

# Flask App
app = Flask(__name__)
app.static_folder = 'static'

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/get")
def get_bot_response():
    userText = request.args.get('msg')
    sentiment = sentiment_analysis(userText)
    response = chatbot_response(userText)
    return f"{response} (Sentiment: {sentiment})"

if __name__ == "__main__":
    app.run(debug=True)

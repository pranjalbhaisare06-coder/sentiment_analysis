
import pickle
from flask import Flask, request, jsonify
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# It's important to make sure nltk stopwords are downloaded for the app to run
import nltk
try:
    stopwords.words('english')
except LookupError:
    nltk.download('stopwords')

app = Flask(__name__)

# Load the model and vectorizer
with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('vectorizer.pkl', 'rb') as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

# Initialize the stemmer
stremmer = PorterStemmer()

# Preprocessing function (needs to match the one used during training)
def stemming(content):
    stemmed_content = re.sub('[^a-zA-Z]',' ',content) # removing not a-z and A-Z
    stemmed_content = stemmed_content.lower()
    stemmed_content = stemmed_content.split()
    stemmed_content = [stremmer.stem(word) for word in stemmed_content if not word in stopwords.words('english')]
    stemmed_content = ' '.join(stemmed_content)
    return stemmed_content

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    text = data.get('text') if isinstance(data, dict) else None
    if not isinstance(text, str) or not text.strip():
        return jsonify({'error': 'text is required'}), 400

    # Preprocess the input text
    processed_text = stemming(text)
    
    # Transform the text using the loaded vectorizer
    transformed_text = vectorizer.transform([processed_text])

    # Make prediction
    sentiment = model.predict(transformed_text)

    if sentiment == 0:
        prediction = "Negative"
    else:
        prediction = "Positive"

    return jsonify({'sentiment': prediction})

@app.route('/')
def home():
    return "Sentiment Analysis API. Send a POST request to /predict with JSON data {'text': 'your text here'}"

if __name__ == '__main__':
    # For local development, use app.run(debug=True)
    # For deployment, consider using a production-ready server like Gunicorn
    app.run(host='0.0.0.0', port=5000)

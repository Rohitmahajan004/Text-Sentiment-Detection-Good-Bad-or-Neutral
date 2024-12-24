from flask import Flask, render_template, request
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

# Initialize the Flask app
app = Flask(__name__)

# Download the VADER lexicon (if not already downloaded)
nltk.download('vader_lexicon')

# Initialize SentimentIntensityAnalyzer
sia = SentimentIntensityAnalyzer()

# Home route
@app.route('/')
def home():
    return render_template('index.html')

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    # Get the input text from the form
    text = request.form['text']
    
    # Get sentiment scores
    sentiment_scores = sia.polarity_scores(text)
    
    # Classify the sentiment based on compound score
    if sentiment_scores['compound'] >= 0.05:
        sentiment = "Good"
    elif sentiment_scores['compound'] <= -0.05:
        sentiment = "Bad"
    else:
        sentiment = "Neutral"

    # Return the prediction result and the input text
    return render_template('index.html', prediction=sentiment, user_text=text)

# Run the app
if __name__ == '__main__':
    app.run(debug=True)

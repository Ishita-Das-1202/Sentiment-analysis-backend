from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import nltk
import os
from fastapi.middleware.cors import CORSMiddleware

nltk.download('stopwords')
nltk.download('wordnet')

# Initialize FastAPI
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# Define input data model
class ReviewInput(BaseModel):
    reviews: list[str]

# Utility functions for text cleaning and preprocessing
def preprocess_text(texts):
    stemmer = PorterStemmer()
    stop_words = set(stopwords.words('english'))
    cleaned_texts = []
    for text in texts:
        text = re.sub('[^a-zA-Z]', ' ', text)
        text = text.lower()
        text = text.split()
        text = [stemmer.stem(word) for word in text if word not in stop_words]
        cleaned_texts.append(' '.join(text))
    return cleaned_texts

# Check if the model and vectorizer are already trained and saved
MODEL_PATH = "model.pkl"
VECTORIZER_PATH = "vectorizer.pkl"

if not os.path.exists(MODEL_PATH) or not os.path.exists(VECTORIZER_PATH):
    # Load the dataset
    data_path = 'reviews.tsv'
    if not os.path.exists(data_path):
        raise FileNotFoundError("Dataset file 'reviews.tsv' not found.")

    data = pd.read_csv(data_path, sep='\t')

    # Preprocess the reviews
    cleaned_reviews = preprocess_text(data['Review'].tolist())

    # Vectorize the data
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(cleaned_reviews)
    y = data['Liked']

    # Train the model
    model = MultinomialNB()
    model.fit(X, y)

    # Save the model and vectorizer
    joblib.dump(model, MODEL_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)
else:
    # Load the pre-trained model and vectorizer
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)

# Define the prediction endpoint
@app.post("/predict")
async def predict_sentiment(input_data: ReviewInput):
    try:
        # Preprocess the input reviews
        cleaned_reviews = preprocess_text(input_data.reviews)

        # Transform the reviews using the vectorizer
        X_new = vectorizer.transform(cleaned_reviews)

        # Predict the sentiment
        predictions = model.predict(X_new)

        # Format the response
        response = {
            "results": [
                {"review": input_data.reviews[i], "prediction": int(predictions[i])}
                for i in range(len(predictions))
            ]
        }
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

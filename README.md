# Sentiment-analysis-backend

### Instruction
pip install fastapi uvicorn scikit-learn pandas numpy nltk joblib wordcloud matplotlib seaborn

uvicorn main:app --reload


curl -X POST "http://127.0.0.1:8000/predict" \
-H "Content-Type: application/json" \
-d '{
  "reviews": [
    "The food was amazing and the service was excellent!",
    "I did not enjoy the meal, it was too salty.",
    "Absolutely fantastic experience, highly recommend!",
    "Terrible service and the food was cold."
  ]
}'


{
  "results": [
    {"review": "The food was amazing and the service was excellent!", "prediction": 1},
    {"review": "I did not enjoy the meal, it was too salty.", "prediction": 0},
    {"review": "Absolutely fantastic experience, highly recommend!", "prediction": 1},
    {"review": "Terrible service and the food was cold.", "prediction": 0}
  ]
}

# Medication Adherence API

Predict medication adherence using DAG Model.

## Endpoints
- GET /health - Check API status
- GET /info - Model information  
- GET /features - Required features
- POST /predict - Make predictions

## Usage
```python
import requests
response = requests.post("https://your-app.onrender.com/predict", 
                       json={"features": [[your, features, here]]})
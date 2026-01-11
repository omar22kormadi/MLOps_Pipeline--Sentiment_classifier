# api/app.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import re
from typing import List

app = FastAPI(
    title="Sentiment Analysis API",
    description="GoEmotions-based sentiment classifier (6 emotions)",
    version="1.0.0"
)

# Load model at startup
try:
    model = joblib.load('models/zenml_sentiment_model.pkl')
    vectorizer = joblib.load('models/zenml_tfidf_vectorizer.pkl')
    print("✅ Model loaded successfully")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None
    vectorizer = None

class TextRequest(BaseModel):
    text: str

class PredictionResponse(BaseModel):
    text: str
    predicted_emotion: str
    confidence: float

class BatchRequest(BaseModel):
    texts: List[str]

def preprocess(text: str) -> str:
    """Preprocess text (same as training)"""
    return re.sub(r'\s+', ' ', text.lower())

@app.get("/")
def root():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "emotions": ["anger", "fear", "joy", "love", "sadness", "surprise"]
    }

@app.post("/predict", response_model=PredictionResponse)
def predict(request: TextRequest):
    """Predict emotion for a single text"""
    if model is None or vectorizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Preprocess and predict
    text_clean = preprocess(request.text)
    X = vectorizer.transform([text_clean])
    prediction = model.predict(X)[0]
    probabilities = model.predict_proba(X)[0]
    confidence = float(max(probabilities))
    
    return PredictionResponse(
        text=request.text,
        predicted_emotion=prediction,
        confidence=confidence
    )

@app.post("/predict/batch")
def predict_batch(request: BatchRequest):
    """Predict emotions for multiple texts"""
    if model is None or vectorizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    texts_clean = [preprocess(text) for text in request.texts]
    X = vectorizer.transform(texts_clean)
    predictions = model.predict(X)
    probabilities = model.predict_proba(X)
    
    results = []
    for i, text in enumerate(request.texts):
        results.append({
            "text": text,
            "predicted_emotion": predictions[i],
            "confidence": float(max(probabilities[i]))
        })
    
    return {"predictions": results}

@app.get("/health")
def health():
    """Kubernetes health check"""
    return {"status": "ok"}

# api/app.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import re
import os
from typing import List


app = FastAPI(
    title="Sentiment Analysis API",
    description="GoEmotions-based sentiment classifier (6 emotions)",
    version="2.0.0"
)


# Load model version from environment variable (default: v1)
MODEL_VERSION = os.getenv("MODEL_VERSION", "v1")

# Load appropriate model
try:
    if MODEL_VERSION == "v2":
        model = joblib.load('models/sentiment_model_v2.pkl')
        vectorizer = joblib.load('models/tfidf_vectorizer_v2.pkl')
        print(f"✅ V2 Model loaded (Test Accuracy: 85.99%)")
    else:
        # Try ZenML models first, fallback to v1
        try:
            model = joblib.load('models/zenml_sentiment_model.pkl')
            vectorizer = joblib.load('models/zenml_tfidf_vectorizer.pkl')
            print(f"✅ ZenML Model loaded")
        except:
            model = joblib.load('models/sentiment_model.pkl')
            vectorizer = joblib.load('models/tfidf_vectorizer.pkl')
            print(f"✅ V1 Model loaded (Test Accuracy: 87.78%)")
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
    model_version: str


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
        "model_version": MODEL_VERSION,
        "accuracy": "87.78%" if MODEL_VERSION == "v1" else "85.99%",
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
        confidence=confidence,
        model_version=MODEL_VERSION
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
            "confidence": float(max(probabilities[i])),
            "model_version": MODEL_VERSION
        })
    
    return {"predictions": results}


@app.get("/health")
def health():
    """Kubernetes health check"""
    return {
        "status": "ok",
        "model_version": MODEL_VERSION
    }

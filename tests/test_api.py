# tests/test_api.py
import pytest
from fastapi.testclient import TestClient
from api.app import app

client = TestClient(app)

def test_health_check():
    """Test API health endpoint"""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_predict_joy():
    """Test joy prediction"""
    response = client.post(
        "/predict",
        json={"text": "I am so happy today!"}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["predicted_emotion"] == "joy"
    assert data["confidence"] > 0.5

# 🎭 Sentiment Analysis MLOps Pipeline

End-to-End MLOps project for emotion classification (6 emotions: **joy, sadness, anger, fear, love, surprise**).

---

## 🌐 Live Deployments

### 🎨 Gradio Web Interface (UI)

Beautiful, user-friendly interface for testing the model:

- **URL**: http://sentiment-ui-omar.spaincentral.azurecontainer.io:8000

### 🔌 FastAPI REST API

Developer-friendly JSON API with Swagger/OpenAPI docs:

- **Docs**: http://sentiment-omar.spaincentral.azurecontainer.io:8000/docs

---

## 📊 Project Overview

This project builds and serves a sentiment/emotion classifier using a complete MLOps toolchain:

- Logistic Regression classifier with TF‑IDF features
- Separate **v1** and **v2** models (different hyperparameters and feature spaces)
- Local training, tracking, containerization, then deployment on Azure Container Instances (Spain Central)

---

## 🛠️ Tech & MLOps Stack

| Layer                  | Tool / Library                |
|------------------------|-------------------------------|
| Programming            | Python 3.10                   |
| Modeling               | scikit-learn (LogisticRegression + TF‑IDF) |
| Data Versioning        | DVC                           |
| Remote Storage         | Azure Blob Storage            |
| Experiment Tracking    | MLflow                        |
| Hyperparameter Tuning  | Optuna                        |
| Pipelines              | ZenML                         |
| API                    | FastAPI                       |
| UI                     | Gradio                        |
| Containerization       | Docker, Docker Compose        |
| Cloud Deploy           | Azure Container Instances     |
| CI/CD                  | Git + GitLab                  |

---

## 📁 Project Structure

```
sentiment-mlops/
├── .dvc/                     # DVC internal config
├── .zen/                     # ZenML config
├── api/
│   └── app.py                # FastAPI app (v1/v2 switch via env var)
├── data/
│   ├── train.txt             # Training data (DVC-tracked)
│   ├── test.txt              # Test data (DVC-tracked)
│   └── val.txt               # Validation data (DVC-tracked)
├── mlruns/                   # MLflow tracking directory
├── models/
│   ├── sentiment_model.pkl           # v1 model
│   ├── tfidf_vectorizer.pkl          # v1 vectorizer
│   ├── sentiment_model_v2.pkl        # v2 model
│   ├── tfidf_vectorizer_v2.pkl       # v2 vectorizer
│   └── zenml_*                       # ZenML-produced artifacts
├── pipelines/
│   ├── __init__.py
│   └── training_pipeline.py  # ZenML training pipeline
├── steps/
│   ├── __init__.py
│   ├── data_steps.py         # Data loading / preprocessing step
│   └── model_steps.py        # Training / evaluation step
├── tests/
│   ├── __init__.py
│   └── test_api.py           # Basic API tests
├── .gitignore
├── .dvcignore
├── .gitlab-ci.yml            # CI/CD config (build + tests + Docker)
├── Dockerfile                # FastAPI image
├── Dockerfile.gradio         # Gradio UI image
├── docker-compose.yml        # Local multi-service setup (v1 / v2)
├── app_gradio.py             # Gradio interface (local & Azure UI)
├── requirements.txt          # Python dependencies
├── src/
│   ├── train.py              # Baseline training (v1)
│   ├── train_v2.py           # Optimized training (v2)
│   └── optuna_hpo.py         # Hyperparameter search with Optuna
└── README.md
```

---

## 🚀 Quickstart (Local)

### 1. Clone the repository

```bash
git clone <YOUR_REPO_URL>
cd sentiment-mlops
```

### 2. Create virtual environment & install dependencies

```bash
python -m venv .venv
source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Pull data with DVC

```bash
dvc pull
```

This fetches `train.txt`, `val.txt`, `test.txt` from the configured Azure Blob remote.

### 4. Train models

```bash
# Train v1
python src/train.py

# Train v2 (optimized TF‑IDF + Logistic Regression)
python src/train_v2.py
```

Artifacts (models, vectorizers, metrics) are saved under `models/` and logged in MLflow.

---

## 📈 Experiments & MLflow

Start MLflow UI locally:

```bash
mlflow ui --backend-store-uri mlruns
```

Then open:

- http://127.0.0.1:5000

You will see runs for:

- Baseline model (v1)
- Optimized model (v2, tuned with Optuna)

---

## 🔄 ZenML Pipeline

The training is also orchestrated with ZenML.

Initialize ZenML (once):

```bash
zenml init
```

Run the training pipeline:

```bash
python pipelines/training_pipeline.py
```

This executes:

1. Data loading step (`data_steps.py`)
2. Training & evaluation step (`model_steps.py`)
3. Logging metrics and saving the best model

---

## 🧪 FastAPI Service (Local)

Run the FastAPI app directly:

```bash
uvicorn api.app:app --reload --host 0.0.0.0 --port 8000
```

Then open:

- Docs: http://127.0.0.1:8000/docs
- Health: GET http://127.0.0.1:8000/
- Predict: POST `/predict` with JSON body:
  ```json
  { "text": "I am very happy today!" }
  ```

Environment variable `MODEL_VERSION` controls which model is loaded:

- `MODEL_VERSION=v1` → legacy model
- `MODEL_VERSION=v2` → optimized model

---

## 🧱 Docker & Docker Compose

### Build images

```bash
# FastAPI API image
docker build -t sentiment-classifier:v1 .

# Gradio UI image
docker build -f Dockerfile.gradio -t sentiment-ui:gradio .
```

### Run locally with Docker

```bash
# FastAPI
docker run -p 8000:8000 -e MODEL_VERSION=v2 sentiment-classifier:v1

# Gradio UI
docker run -p 7860:8000 -e MODEL_VERSION=v2 sentiment-ui:gradio
```

Open:

- API: http://localhost:8000/docs
- UI: http://localhost:7860

### Docker Compose (v1 and v2)

`docker-compose.yml` allows running v1 and v2 simultaneously on different ports (e.g., `8001` and `8002`) to demonstrate upgrade and rollback.

```bash
docker-compose up --build
```

---

## ☁️ Azure Deployment

The project is deployed to Azure Container Instances (region: Spain Central):

- FastAPI API:
  - `az container create` with image `omarkr123/sentiment-api:v2`
  - Public endpoint with Swagger docs at  
    http://sentiment-omar.spaincentral.azurecontainer.io:8000/docs

- Gradio UI:
  - `az container create` with image `omarkr123/sentiment-ui:gradio`
  - Public interface at  
    http://sentiment-ui-omar.spaincentral.azurecontainer.io:8000

Both containers are configured with:

- `--os-type Linux`
- `--environment-variables MODEL_VERSION=v2`
- Public IP and DNS name labels.

---

## ✅ Implemented Features (Checklist)

- [x] Data versioning with DVC + Azure Blob
- [x] Training scripts for v1 and v2
- [x] Hyperparameter optimization with Optuna
- [x] Experiment tracking with MLflow
- [x] ZenML training pipeline (data + model steps)
- [x] FastAPI REST API
- [x] Dockerized API and UI
- [x] Local Docker Compose setup (v1/v2 and rollback)
- [x] Azure Container Instances deployment (API + UI)
- [x] Gradio professional interface

---

## 📌 Usage Examples

### Call the API (remote)

```bash
# Health
curl http://sentiment-omar.spaincentral.azurecontainer.io:8000/

# Single prediction
curl -X POST "http://sentiment-omar.spaincentral.azurecontainer.io:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"text": "I am so happy today!"}'
```

### Use the UI (remote)

Open in browser:

- http://sentiment-ui-omar.spaincentral.azurecontainer.io:8000

Try sample sentences and inspect predicted emotions and confidence distribution.

---

## 👤 Author

- **Names**: Omar Kormadi and Khadija Ammar
- **Project**: Mini-projet MLOps – Sentiment / Emotion Classifier
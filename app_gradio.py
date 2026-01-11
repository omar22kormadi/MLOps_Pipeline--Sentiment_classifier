# app_gradio.py
import gradio as gr
import joblib
import re
import os

# Load model
MODEL_VERSION = os.getenv("MODEL_VERSION", "v2")

try:
    if MODEL_VERSION == "v2":
        model = joblib.load('models/sentiment_model_v2.pkl')
        vectorizer = joblib.load('models/tfidf_vectorizer_v2.pkl')
        print("✅ V2 Model loaded (Test Accuracy: 85.99%)")
        train_acc, val_acc, test_acc = 0.9128, 0.8689, 0.8599
    else:
        model = joblib.load('models/sentiment_model.pkl')
        vectorizer = joblib.load('models/tfidf_vectorizer.pkl')
        print("✅ V1 Model loaded (Test Accuracy: 87.78%)")
        train_acc, val_acc, test_acc = 0.8778, 0.8500, 0.8778
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None
    vectorizer = None

def preprocess(text):
    """Preprocess text"""
    return re.sub(r'\s+', ' ', text.lower())

def predict_sentiment(text):
    """Predict emotion"""
    if not text or not text.strip():
        return "No text provided", {}
    
    if model is None or vectorizer is None:
        return "Model not loaded", {"error": 1.0}
    
    try:
        text_clean = preprocess(text)
        X = vectorizer.transform([text_clean])
        prediction = model.predict(X)[0]
        probabilities = model.predict_proba(X)[0]
        
        # Create confidence dict
        label_map = dict(zip(model.classes_, probabilities))
        
        return prediction, label_map
    except Exception as e:
        return f"Error: {str(e)}", {}

# CSS Styling
css = """
.gradio-container {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    font-family: 'Arial', sans-serif;
}
h1 {
    color: white !important;
    text-align: center;
    font-size: 2.8em;
    text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
}
.subtitle {
    text-align: center;
    color: white;
    font-size: 1.3em;
    margin-bottom: 30px;
}
"""

# Build Gradio Interface
with gr.Blocks(css=css, title="🎭 Sentiment Classifier Pro") as app:
    gr.HTML(f"""
    <h1>🎭 Emotion Classification Engine</h1>
    <p class='subtitle'>
        ✨ Model v{MODEL_VERSION.upper()} | Accuracy: {test_acc:.2%} | Powered by Azure ☁️
    </p>
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            text_input = gr.Textbox(
                label="💬 Enter Your Text",
                placeholder="Type something like: I feel so happy today!",
                lines=5,
                elem_classes=["input-text"]
            )
            predict_btn = gr.Button("🔮 Analyze Emotion", variant="primary", size="lg")
        
        with gr.Column(scale=1):
            sentiment_output = gr.Label(label="🎯 Predicted Emotion", num_top_classes=6)
            conf_output = gr.Label(label="📊 Confidence Distribution", num_top_classes=6)
    
    with gr.Row():
        gr.Examples(
            examples=[
                ["I love this so much! Best day ever!"],
                ["I'm feeling really sad and lonely today."],
                ["This makes me so angry and frustrated!"],
                ["I'm scared about what might happen."],
                ["What a pleasant surprise! I didn't expect this!"],
                ["I'm surprised !!"]
            ],
            inputs=text_input,
            label="💡 Try These Examples"
        )
    
    # Model Info Tab
    with gr.Tab("📈 Model Performance"):
        gr.Markdown(f"""
        ## 🚀 Deployment Information
        
        | Feature | Value |
        |---------|-------|
        | **Model Version** | v{MODEL_VERSION.upper()} |
        | **Deployment** | Azure Container Instances |
        | **Region** | Spain Central 🇪🇸 |
        | **URL** | sentiment-omar.spaincentral.azurecontainer.io |
        
        ## 📊 Accuracy Metrics
        
        | Dataset Split | Accuracy |
        |---------------|----------|
        | **Training**  | {train_acc:.2%}  |
        | **Validation**| {val_acc:.2%}  |
        | **Test**      | {test_acc:.2%}  |
        
        ## 🎯 Supported Emotions
        - 😊 **Joy** - Happiness, delight, pleasure
        - 😢 **Sadness** - Sorrow, grief, melancholy
        - 😠 **Anger** - Rage, frustration, annoyance
        - 😨 **Fear** - Anxiety, worry, terror
        - ❤️ **Love** - Affection, adoration, care
        - 😲 **Surprise** - Shock, amazement, astonishment
        
        ## 🛠️ Technical Stack
        - **Algorithm**: Logistic Regression + TF-IDF
        - **Features**: 10,000 (unigrams + bigrams)
        - **MLOps**: DVC + MLflow + ZenML + Optuna + Docker
        - **Framework**: Gradio + FastAPI
        """)
    
    # About Tab
    with gr.Tab("ℹ️ About"):
        gr.Markdown("""
        ## 🎓 MLOps Project
        
        This is a complete end-to-end MLOps pipeline for emotion classification:
        
        ### ✅ Features Implemented
        - ✅ Data versioning with **DVC** (Azure Blob Storage)
        - ✅ Experiment tracking with **MLflow**
        - ✅ Hyperparameter optimization with **Optuna**
        - ✅ Pipeline orchestration with **ZenML**
        - ✅ Containerization with **Docker**
        - ✅ CI/CD with **GitLab**
        - ✅ Model versioning (v1 ↔ v2 rollback)
        - ✅ Cloud deployment on **Azure**
        
        ### 👨‍💻 Author
        Omar Kormadi | MLOps & NLP Project 2025-2026
        
        ### 📧 Contact
        For questions or collaborations, reach out via the project repository.
        """)
    
    # Connect button to prediction
    predict_btn.click(
        fn=predict_sentiment,
        inputs=text_input,
        outputs=[sentiment_output, conf_output]
    )

# Launch app
if __name__ == "__main__":
    app.launch(
        server_name="0.0.0.0",
        server_port=8000,  # Same port as FastAPI
        share=False
    )

# run_pipeline.py
from src.pipelines.training_pipeline import sentiment_training_pipeline

if __name__ == "__main__":
    print("🚀 Starting ZenML Sentiment Pipeline...\n")
    
    # Run with Optuna-optimized hyperparameters
    pipeline_run = sentiment_training_pipeline(
        max_features=4000,
        C=7.114,
        max_iter=1500
    )
    
    print("\n✅ Pipeline execution completed!")
    print("📊 Check MLflow UI: http://localhost:5000")

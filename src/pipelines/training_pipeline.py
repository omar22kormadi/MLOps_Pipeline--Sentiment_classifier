# src/pipelines/training_pipeline.py
from zenml import pipeline
from src.steps.data_steps import load_data_step, preprocess_step
from src.steps.model_steps import train_model_step, evaluate_model_step, save_model_step

@pipeline
def sentiment_training_pipeline(
    max_features: int = 4000,
    C: float = 7.11,
    max_iter: int = 1500
):
    """
    End-to-end sentiment classification pipeline
    
    Steps:
    1. Load data from DVC
    2. Preprocess text
    3. Train model with Optuna-optimized params
    4. Evaluate on test set
    5. Save model artifacts
    """
    # Load data
    train_df, test_df, val_df = load_data_step()
    
    # Preprocess
    train_df, test_df, val_df = preprocess_step(train_df, test_df, val_df)
    
    # Train
    model, vectorizer = train_model_step(
        train_df=train_df,
        val_df=val_df,
        max_features=max_features,
        C=C,
        max_iter=max_iter
    )
    
    # Evaluate
    test_accuracy = evaluate_model_step(
        model=model,
        vectorizer=vectorizer,
        test_df=test_df
    )
    
    # Save
    save_model_step(
        model=model,
        vectorizer=vectorizer,
        test_accuracy=test_accuracy
    )

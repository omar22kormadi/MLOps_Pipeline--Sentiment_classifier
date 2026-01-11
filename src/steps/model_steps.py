# src/steps/model_steps.py
from zenml import step
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report
import joblib
from typing import Tuple, Annotated
import numpy as np

@step(enable_cache=False)
def train_model_step(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    max_features: int = 4000,
    C: float = 7.11,
    max_iter: int = 1500
) -> Tuple[
    Annotated[object, "model"],
    Annotated[object, "vectorizer"]
]:
    """Train sentiment classifier with best Optuna hyperparameters"""
    
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("zenml-sentiment-pipeline")
    
    with mlflow.start_run(run_name="zenml-pipeline-run"):
        print(f"🚂 Training with: max_features={max_features}, C={C}, max_iter={max_iter}")
        
        # Log params
        mlflow.log_param("max_features", max_features)
        mlflow.log_param("C", C)
        mlflow.log_param("max_iter", max_iter)
        mlflow.log_param("pipeline", "zenml")
        
        # Vectorize
        vectorizer = TfidfVectorizer(max_features=max_features, stop_words='english')
        X_train = vectorizer.fit_transform(train_df['text'])
        X_val = vectorizer.transform(val_df['text'])
        y_train, y_val = train_df['label'], val_df['label']
        
        # Train
        model = LogisticRegression(C=C, max_iter=max_iter, solver='lbfgs')
        model.fit(X_train, y_train)
        
        # Validate
        val_acc = accuracy_score(y_val, model.predict(X_val))
        val_f1 = f1_score(y_val, model.predict(X_val), average='macro')
        
        mlflow.log_metric("val_accuracy", val_acc)
        mlflow.log_metric("val_f1_macro", val_f1)
        
        print(f"✅ Training complete: val_acc={val_acc:.4f}, val_f1={val_f1:.4f}")
        
        # Save artifacts
        mlflow.sklearn.log_model(model, "model")
    
    return model, vectorizer

@step(enable_cache=False)
def evaluate_model_step(
    model: object,
    vectorizer: object,
    test_df: pd.DataFrame
) -> Annotated[float, "test_accuracy"]:
    """Evaluate model on test set"""
    
    print("📊 Evaluating on test set...")
    
    X_test = vectorizer.transform(test_df['text'])
    y_test = test_df['label']
    y_pred = model.predict(X_test)
    
    test_acc = accuracy_score(y_test, y_pred)
    test_f1 = f1_score(y_test, y_pred, average='macro')
    
    # Log to MLflow
    mlflow.log_metric("test_accuracy", test_acc)
    mlflow.log_metric("test_f1_macro", test_f1)
    
    print(f"✅ Test Accuracy: {test_acc:.4f}")
    print(f"✅ Test F1: {test_f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    return test_acc

@step
def save_model_step(
    model: object,
    vectorizer: object,
    test_accuracy: float
) -> None:
    """Save final model artifacts"""
    
    print(f"💾 Saving model (test_acc={test_accuracy:.4f})...")
    
    joblib.dump(model, 'models/zenml_sentiment_model.pkl')
    joblib.dump(vectorizer, 'models/zenml_tfidf_vectorizer.pkl')
    
    # Save metadata
    with open('models/model_metadata.txt', 'w') as f:
        f.write(f"Test Accuracy: {test_accuracy:.4f}\n")
        f.write(f"Pipeline: ZenML\n")
        f.write(f"Timestamp: {pd.Timestamp.now()}\n")
    
    print("✅ Model saved to models/")

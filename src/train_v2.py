# src/train_v2.py
import pandas as pd
import re
import mlflow
import mlflow.sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib


# Set MLflow tracking
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("sentiment-goemotions")


def load_data(filename):
    data = []
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if ';' in line:
                text, label = line.split(';', 1)
                label = label.strip().lower()
                if label in ['joy', 'sadness', 'anger', 'fear', 'love', 'surprise']:
                    data.append({'text': text.strip(), 'label': label})
    return pd.DataFrame(data)


def preprocess(text):
    return re.sub(r'\s+', ' ', text.lower())


# Load data
train_df = load_data('data/train.txt')
test_df = load_data('data/test.txt')
val_df = load_data('data/val.txt')


train_df['text'] = train_df['text'].apply(preprocess)
test_df['text'] = test_df['text'].apply(preprocess)
val_df['text'] = val_df['text'].apply(preprocess)


# Start MLflow run
with mlflow.start_run(run_name="v2-optimized-tfidf-lr"):
    
    # CHANGED PARAMETERS FOR V2
    max_features = 10000  # CHANGED from 5000
    max_iter = 2000       # CHANGED from 1000
    C = 0.5               # NEW: Regularization strength
    
    mlflow.log_param("max_features", max_features)
    mlflow.log_param("max_iter", max_iter)
    mlflow.log_param("C", C)
    mlflow.log_param("model_type", "LogisticRegression")
    mlflow.log_param("vectorizer", "TfidfVectorizer")
    mlflow.log_param("version", "v2.0")
    
    # Vectorize with MORE features
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        stop_words='english',
        ngram_range=(1, 2)  # NEW: Add bigrams
    )
    X_train = vectorizer.fit_transform(train_df['text'])
    X_test = vectorizer.transform(test_df['text'])
    X_val = vectorizer.transform(val_df['text'])
    y_train, y_test, y_val = train_df['label'], test_df['label'], val_df['label']
    
    # Train with DIFFERENT params
    model = LogisticRegression(
        max_iter=max_iter,
        C=C,  # NEW parameter
        multi_class='multinomial',
        solver='lbfgs'
    )
    model.fit(X_train, y_train)
    
    # Evaluate
    train_acc = accuracy_score(y_train, model.predict(X_train))
    val_acc = accuracy_score(y_val, model.predict(X_val))
    test_acc = accuracy_score(y_test, model.predict(X_test))
    
    # Log metrics
    mlflow.log_metric("train_accuracy", train_acc)
    mlflow.log_metric("val_accuracy", val_acc)
    mlflow.log_metric("test_accuracy", test_acc)
    
    print(f"✅ V2 - Train: {train_acc:.4f}, Val: {val_acc:.4f}, Test: {test_acc:.4f}")
    
    # Confusion matrix
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens')  # Green for v2
    plt.title('Confusion Matrix - V2.0')
    plt.ylabel('True')
    plt.xlabel('Predicted')
    plt.savefig('confusion_matrix_v2.png')
    plt.close()
    
    # Log artifacts
    mlflow.log_artifact('confusion_matrix_v2.png')
    mlflow.sklearn.log_model(model, "model")
    
    # Save as v2
    joblib.dump(model, 'models/sentiment_model_v2.pkl')
    joblib.dump(vectorizer, 'models/tfidf_vectorizer_v2.pkl')
    
    print("✅ V2 model saved in models/sentiment_model_v2.pkl")

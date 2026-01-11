# src/optuna_hpo.py
import pandas as pd
import re
import optuna
import mlflow
import mlflow.sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
import joblib

# MLflow setup
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("sentiment-optuna-hpo")

def load_data(filename):
    data = []
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            if ';' in line:
                text, label = line.split(';', 1)
                label = label.strip().lower()
                if label in ['joy', 'sadness', 'anger', 'fear', 'love', 'surprise']:
                    data.append({'text': text.strip(), 'label': label})
    return pd.DataFrame(data)

def preprocess(text):
    return re.sub(r'\s+', ' ', text.lower())

# Load data
print("Loading data...")
train_df = load_data('data/train.txt')
val_df = load_data('data/val.txt')
test_df = load_data('data/test.txt')

train_df['text'] = train_df['text'].apply(preprocess)
val_df['text'] = val_df['text'].apply(preprocess)
test_df['text'] = test_df['text'].apply(preprocess)

# Optuna objective function
def objective(trial):
    # Suggest hyperparameters
    max_features = trial.suggest_int('max_features', 1000, 10000, step=1000)
    C = trial.suggest_float('C', 0.01, 10.0, log=True)
    max_iter = trial.suggest_int('max_iter', 500, 2000, step=500)
    
    with mlflow.start_run(run_name=f"optuna-trial-{trial.number}", nested=True):
        # Log params
        mlflow.log_param("max_features", max_features)
        mlflow.log_param("C", C)
        mlflow.log_param("max_iter", max_iter)
        mlflow.log_param("trial_number", trial.number)
        
        # Vectorize
        vectorizer = TfidfVectorizer(max_features=max_features, stop_words='english')
        X_train = vectorizer.fit_transform(train_df['text'])
        X_val = vectorizer.transform(val_df['text'])
        y_train, y_val = train_df['label'], val_df['label']
        
        # Train
        model = LogisticRegression(C=C, max_iter=max_iter, multi_class='multinomial', solver='lbfgs')
        model.fit(X_train, y_train)
        
        # Evaluate on validation
        y_pred = model.predict(X_val)
        val_acc = accuracy_score(y_val, y_pred)
        val_f1 = f1_score(y_val, y_pred, average='macro')
        
        # Log metrics
        mlflow.log_metric("val_accuracy", val_acc)
        mlflow.log_metric("val_f1_macro", val_f1)
        
        print(f"Trial {trial.number}: val_acc={val_acc:.4f}, val_f1={val_f1:.4f}")
        
    return val_f1  # Optimize F1 score

# Run Optuna study
if __name__ == "__main__":
    print("Starting Optuna optimization...")
    
    with mlflow.start_run(run_name="optuna-study-parent"):
        study = optuna.create_study(
            direction='maximize',
            study_name='sentiment-hpo',
            sampler=optuna.samplers.TPESampler(seed=42)
        )
        
        study.optimize(objective, n_trials=10)  # 10 trials
        
        # Best params
        print("\n" + "="*50)
        print("Best trial:")
        print(f"  Value (F1): {study.best_trial.value:.4f}")
        print(f"  Params: {study.best_trial.params}")
        
        # Log best params to MLflow
        mlflow.log_params(study.best_trial.params)
        mlflow.log_metric("best_val_f1", study.best_trial.value)
        
        # Retrain with best params on full train set
        print("\nRetraining with best params...")
        best_params = study.best_trial.params
        
        vectorizer = TfidfVectorizer(max_features=best_params['max_features'], stop_words='english')
        X_train = vectorizer.fit_transform(train_df['text'])
        X_test = vectorizer.transform(test_df['text'])
        y_train, y_test = train_df['label'], test_df['label']
        
        best_model = LogisticRegression(
            C=best_params['C'],
            max_iter=best_params['max_iter'],
            multi_class='multinomial',
            solver='lbfgs'
        )
        best_model.fit(X_train, y_train)
        
        # Test accuracy
        test_acc = accuracy_score(y_test, best_model.predict(X_test))
        test_f1 = f1_score(y_test, best_model.predict(X_test), average='macro')
        
        mlflow.log_metric("test_accuracy", test_acc)
        mlflow.log_metric("test_f1_macro", test_f1)
        
        print(f"\nFinal Test Accuracy: {test_acc:.4f}")
        print(f"Final Test F1: {test_f1:.4f}")
        
        # Save best model
        joblib.dump(best_model, 'models/best_optuna_model.pkl')
        joblib.dump(vectorizer, 'models/best_optuna_vectorizer.pkl')
        mlflow.sklearn.log_model(best_model, "best_model")
        
        print("✅ Optuna optimization completed!")

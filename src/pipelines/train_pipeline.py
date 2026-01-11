from zenml import step, pipeline
from sklearn.linear_model import LogisticRegression
from zenml.artifacts import ModelArtifact

@step
def load_data() -> tuple:
    # Your load_data + preprocess
    return X_train, X_test, y_train, y_test

@step
def train_model(X_train, y_train) -> ModelArtifact:
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    return model

@pipeline
def sentiment_pipeline():
    X_train, X_test, y_train, y_test = load_data()
    model = train_model(X_train, y_train)
    # eval_step...

if __name__ == "__main__":
    sentiment_pipeline()

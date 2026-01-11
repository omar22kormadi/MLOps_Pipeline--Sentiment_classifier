# src/steps/data_steps.py
from zenml import step
import pandas as pd
import re
from typing import Tuple, Annotated

@step
def load_data_step() -> Tuple[
    Annotated[pd.DataFrame, "train_df"],
    Annotated[pd.DataFrame, "test_df"],
    Annotated[pd.DataFrame, "val_df"]
]:
    """Load train/test/val datasets from DVC-tracked data"""
    def load_file(filename):
        data = []
        with open(filename, 'r', encoding='utf-8') as f:
            for line in f:
                if ';' in line:
                    text, label = line.split(';', 1)
                    label = label.strip().lower()
                    if label in ['joy', 'sadness', 'anger', 'fear', 'love', 'surprise']:
                        data.append({'text': text.strip(), 'label': label})
        return pd.DataFrame(data)
    
    print("📂 Loading datasets...")
    train_df = load_file('data/train.txt')
    test_df = load_file('data/test.txt')
    val_df = load_file('data/val.txt')
    
    print(f"✅ Loaded: Train={len(train_df)}, Test={len(test_df)}, Val={len(val_df)}")
    return train_df, test_df, val_df

@step
def preprocess_step(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    val_df: pd.DataFrame
) -> Tuple[
    Annotated[pd.DataFrame, "train_preprocessed"],
    Annotated[pd.DataFrame, "test_preprocessed"],
    Annotated[pd.DataFrame, "val_preprocessed"]
]:
    """Preprocess text data (lowercase, normalize spaces)"""
    def preprocess(text):
        return re.sub(r'\s+', ' ', text.lower())
    
    print("🧹 Preprocessing text...")
    train_df['text'] = train_df['text'].apply(preprocess)
    test_df['text'] = test_df['text'].apply(preprocess)
    val_df['text'] = val_df['text'].apply(preprocess)
    
    print("✅ Preprocessing complete")
    return train_df, test_df, val_df

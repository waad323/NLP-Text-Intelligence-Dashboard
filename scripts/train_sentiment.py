"""Sentiment Analysis Model Training Pipeline.

Trains a TF-IDF + Logistic Regression classifier on IMDb movie reviews.
Supports command-line arguments, cross-validation, and metrics export.
"""

import argparse
import os
import pickle
import sys
from typing import Optional
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

# Add root directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from scripts.preprocess import clean_text


def train_sentiment_model(
    data_path: Optional[str] = None,
    output_dir: Optional[str] = None,
    max_features: int = 5000,
    test_size: float = 0.2,
    random_state: int = 42,
) -> dict:
    """Trains and serializes the sentiment analysis model pipeline.

    Args:
        data_path: Path to sentiment CSV dataset (must have 'review' and 'sentiment' columns).
        output_dir: Directory to save serialized model and vectorizer.
        max_features: Maximum vocabulary size for TF-IDF.
        test_size: Test split ratio.
        random_state: Reproducibility seed.

    Returns:
        Dictionary of performance metrics.
    """
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

    if data_path is None:
        # Check for sample dataset first, then full dataset
        candidate_sample = os.path.join(root_dir, 'datasets', 'sample_sentiment_data.csv')
        candidate_full = os.path.join(root_dir, 'datasets', 'sentiment_data.csv')
        if os.path.exists(candidate_sample):
            data_path = candidate_sample
        elif os.path.exists(candidate_full):
            data_path = candidate_full
        else:
            raise FileNotFoundError(f"No sentiment dataset found in {os.path.join(root_dir, 'datasets')}")

    if output_dir is None:
        output_dir = os.path.join(root_dir, 'saved_models')

    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading sentiment dataset from: {data_path}")
    df = pd.read_csv(data_path)
    print(f"Loaded {len(df):,} records.")

    # Validate schema
    text_col = 'review' if 'review' in df.columns else df.columns[0]
    label_col = 'sentiment' if 'sentiment' in df.columns else df.columns[1]

    # Preprocessing
    print("Preprocessing text data (cleaning, stopword removal, lemmatization)...")
    df['cleaned'] = df[text_col].apply(clean_text)
    df = df[df['cleaned'].str.strip() != ""].dropna(subset=['cleaned', label_col])
    print(f"Usable records after normalization: {len(df):,}")

    X = df['cleaned']
    y = df[label_col].astype(str).str.lower()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    print(f"Train split: {len(X_train):,} samples | Test split: {len(X_test):,} samples")

    print(f"Extracting TF-IDF features (max_features={max_features}, ngram_range=(1,2))...")
    tfidf = TfidfVectorizer(max_features=max_features, ngram_range=(1, 2), sublinear_tf=True)
    X_train_vec = tfidf.fit_transform(X_train)
    X_test_vec = tfidf.transform(X_test)

    print("Training Logistic Regression classifier (C=1.0, max_iter=1000)...")
    model = LogisticRegression(C=1.0, max_iter=1000, random_state=random_state)
    model.fit(X_train_vec, y_train)

    # Evaluation
    y_pred = model.predict(X_test_vec)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, pos_label='positive', average='binary')
    rec = recall_score(y_test, y_pred, pos_label='positive', average='binary')
    f1 = f1_score(y_test, y_pred, pos_label='positive', average='binary')

    print("\n" + "=" * 50)
    print("      SENTIMENT MODEL EVALUATION METRICS")
    print("=" * 50)
    print(f"  Accuracy  : {acc:.4f} ({acc * 100:.2f}%)")
    print(f"  Precision : {prec:.4f}")
    print(f"  Recall    : {rec:.4f}")
    print(f"  F1 Score  : {f1:.4f}")
    print("-" * 50)
    print(classification_report(y_test, y_pred))
    print("=" * 50 + "\n")

    # Serialize artifacts
    model_path = os.path.join(output_dir, 'sentiment_model.pkl')
    tfidf_path = os.path.join(output_dir, 'sentiment_tfidf.pkl')

    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    with open(tfidf_path, 'wb') as f:
        pickle.dump(tfidf, f)

    print(f"Saved model artifact -> {model_path}")
    print(f"Saved TF-IDF vectorizer -> {tfidf_path}")

    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Sentiment Analysis Model")
    parser.add_argument("--data_path", type=str, default=None, help="Path to sentiment CSV dataset")
    parser.add_argument("--output_dir", type=str, default=None, help="Directory to save model checkpoints")
    parser.add_argument("--max_features", type=int, default=5000, help="Max vocabulary features")
    args = parser.parse_args()

    train_sentiment_model(data_path=args.data_path, output_dir=args.output_dir, max_features=args.max_features)

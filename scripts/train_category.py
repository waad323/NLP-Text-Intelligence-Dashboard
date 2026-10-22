"""Topic Classification Model Training Pipeline.

Trains a multi-class TF-IDF + Logistic Regression classifier on BBC News
(Business, Entertainment, Politics, Sport, Tech).
Supports command-line arguments and unified checkpoint serialization.
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
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from scripts.preprocess import clean_text


def train_category_model(
    data_path: Optional[str] = None,
    output_dir: Optional[str] = None,
    max_features: int = 10000,
    test_size: float = 0.2,
    random_state: int = 42,
) -> dict:
    """Trains and serializes the multi-class BBC news topic classifier.

    Args:
        data_path: Path to BBC classification CSV (must have 'text' and 'labels' columns).
        output_dir: Directory to save serialized model checkpoint.
        max_features: Maximum vocabulary size for TF-IDF.
        test_size: Test split ratio.
        random_state: Reproducibility seed.

    Returns:
        Dictionary of performance metrics.
    """
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

    if data_path is None:
        data_path = os.path.join(root_dir, 'datasets', 'classification_data.csv')

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Classification dataset not found at: {data_path}")

    if output_dir is None:
        output_dir = os.path.join(root_dir, 'saved_models')

    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading BBC News classification dataset from: {data_path}")
    df = pd.read_csv(data_path)
    print(f"Loaded {len(df):,} records.")

    text_col = 'text' if 'text' in df.columns else df.columns[0]
    label_col = 'labels' if 'labels' in df.columns else df.columns[1]

    # Preprocessing
    print("Normalizing text documents...")
    df = df.dropna(subset=[text_col, label_col])
    df['cleaned'] = df[text_col].apply(clean_text)
    df = df[df['cleaned'].str.strip() != ""].dropna(subset=['cleaned', label_col])
    df = df.reset_index(drop=True)
    print(f"Usable records after filtration: {len(df):,}")

    X = df['cleaned']
    y = df[label_col].astype(str).str.lower()

    print("\nClass Distribution:")
    print(y.value_counts())

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    print(f"\nTrain split: {len(X_train):,} samples | Test split: {len(X_test):,} samples")

    print(f"Extracting TF-IDF features (max_features={max_features}, ngram_range=(1,2))...")
    tfidf = TfidfVectorizer(max_features=max_features, ngram_range=(1, 2), sublinear_tf=True)
    X_train_vec = tfidf.fit_transform(X_train)
    X_test_vec = tfidf.transform(X_test)

    print("Training Multi-Class Logistic Regression (max_iter=1000)...")
    model = LogisticRegression(max_iter=1000, random_state=random_state)
    model.fit(X_train_vec, y_train)

    # Evaluation
    y_pred = model.predict(X_test_vec)
    acc = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average='macro')
    f1_weighted = f1_score(y_test, y_pred, average='weighted')

    print("\n" + "=" * 55)
    print("      TOPIC CLASSIFIER EVALUATION REPORT")
    print("=" * 55)
    print(f"  Overall Accuracy : {acc:.4f} ({acc * 100:.2f}%)")
    print(f"  Macro F1 Score   : {f1_macro:.4f}")
    print(f"  Weighted F1 Score: {f1_weighted:.4f}")
    print("-" * 55)
    print(classification_report(y_test, y_pred))
    print("=" * 55 + "\n")

    # Serialize unified model dictionary
    save_path = os.path.join(output_dir, 'text_classifier.pkl')
    checkpoint = {
        'model': model,
        'vectorizer': tfidf,
        'classes': list(model.classes_),
        'metrics': {
            'accuracy': float(acc),
            'f1_macro': float(f1_macro),
            'f1_weighted': float(f1_weighted),
        },
    }
    with open(save_path, 'wb') as f:
        pickle.dump(checkpoint, f)

    print(f"Saved unified model checkpoint -> {save_path}")
    return checkpoint['metrics']


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train BBC News Topic Classifier")
    parser.add_argument("--data_path", type=str, default=None, help="Path to classification CSV")
    parser.add_argument("--output_dir", type=str, default=None, help="Directory to save model checkpoints")
    parser.add_argument("--max_features", type=int, default=10000, help="Max vocabulary features")
    args = parser.parse_args()

    train_category_model(data_path=args.data_path, output_dir=args.output_dir, max_features=args.max_features)

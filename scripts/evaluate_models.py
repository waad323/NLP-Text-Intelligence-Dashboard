"""Model Evaluation and Visualization Benchmark Script.

Evaluates pre-trained Sentiment and BBC News topic classification models,
generates high-resolution confusion matrix heatmaps, and saves evaluation
artifacts for documentation and recruiter review.
"""

import os
import pickle
import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from scripts.preprocess import clean_text


def evaluate_sentiment_model(root_dir: str) -> dict:
    """Evaluates sentiment classifier and saves confusion matrix plot."""
    print("\n--- Evaluating Sentiment Analysis Pipeline ---")
    models_dir = os.path.join(root_dir, 'saved_models')
    model_path = os.path.join(models_dir, 'sentiment_model.pkl')
    tfidf_path = os.path.join(models_dir, 'sentiment_tfidf.pkl')

    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    with open(tfidf_path, 'rb') as f:
        tfidf = pickle.load(f)

    if not hasattr(model, 'multi_class'):
        model.multi_class = 'auto'

    data_sample = os.path.join(root_dir, 'datasets', 'sample_sentiment_data.csv')
    data_full = os.path.join(root_dir, 'datasets', 'sentiment_data.csv')
    data_path = data_sample if os.path.exists(data_sample) else data_full

    df = pd.read_csv(data_path)
    text_col = 'review' if 'review' in df.columns else df.columns[0]
    label_col = 'sentiment' if 'sentiment' in df.columns else df.columns[1]

    # Preprocess
    df['cleaned'] = df[text_col].apply(clean_text)
    df = df[df['cleaned'].str.strip() != ""].dropna(subset=['cleaned', label_col])

    X = df['cleaned']
    y = df[label_col].astype(str).str.lower()

    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    X_test_vec = tfidf.transform(X_test)
    y_pred = model.predict(X_test_vec)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, pos_label='positive', average='binary')
    classes = list(model.classes_)
    cm = confusion_matrix(y_test, y_pred, labels=classes)

    print(f"Sentiment Accuracy: {acc:.4f} ({acc * 100:.2f}%) | F1: {f1:.4f}")

    # Plot Confusion Matrix
    plt.figure(figsize=(6, 5), dpi=150)
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=[c.capitalize() for c in classes],
        yticklabels=[c.capitalize() for c in classes],
        cbar=True,
    )
    plt.title(f'IMDb Sentiment Classifier (Accuracy: {acc * 100:.1f}%)', fontsize=12, fontweight='bold', pad=12)
    plt.xlabel('Predicted Sentiment', fontsize=10, labelpad=8)
    plt.ylabel('Actual Ground Truth', fontsize=10, labelpad=8)
    plt.tight_layout()

    report_path = os.path.join(root_dir, 'reports', 'confusion_matrix_sentiment.png')
    asset_path = os.path.join(root_dir, 'static', 'assets', 'confusion_matrix_sentiment.png')
    plt.savefig(report_path)
    plt.savefig(asset_path)
    plt.close()
    print(f"Saved sentiment confusion matrix -> {report_path}")

    return {"accuracy": acc, "f1": f1, "classes": classes}


def evaluate_category_model(root_dir: str) -> dict:
    """Evaluates 5-class BBC News classifier and saves confusion matrix plot."""
    print("\n--- Evaluating Topic Classification Pipeline ---")
    cat_path = os.path.join(root_dir, 'saved_models', 'text_classifier.pkl')
    with open(cat_path, 'rb') as f:
        checkpoint = pickle.load(f)
        model = checkpoint['model']
        tfidf = checkpoint['vectorizer']

    if not hasattr(model, 'multi_class'):
        model.multi_class = 'auto'

    data_path = os.path.join(root_dir, 'datasets', 'classification_data.csv')
    df = pd.read_csv(data_path).dropna(subset=['text', 'labels'])
    df['cleaned'] = df['text'].apply(clean_text)
    df = df[df['cleaned'].str.strip() != ""].dropna(subset=['cleaned', 'labels'])

    X = df['cleaned']
    y = df['labels'].astype(str).str.lower()

    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    X_test_vec = tfidf.transform(X_test)
    y_pred = model.predict(X_test_vec)

    acc = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average='macro')
    classes = list(model.classes_)
    cm = confusion_matrix(y_test, y_pred, labels=classes)

    print(f"Topic Classifier Accuracy: {acc:.4f} ({acc * 100:.2f}%) | Macro F1: {f1_macro:.4f}")

    # Plot Multi-Class Confusion Matrix
    plt.figure(figsize=(7, 6), dpi=150)
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Purples',
        xticklabels=[c.capitalize() for c in classes],
        yticklabels=[c.capitalize() for c in classes],
        cbar=True,
    )
    plt.title(f'BBC News Topic Classifier (Accuracy: {acc * 100:.1f}%)', fontsize=12, fontweight='bold', pad=12)
    plt.xlabel('Predicted Topic', fontsize=10, labelpad=8)
    plt.ylabel('Actual Category', fontsize=10, labelpad=8)
    plt.tight_layout()

    report_path = os.path.join(root_dir, 'reports', 'confusion_matrix_category.png')
    asset_path = os.path.join(root_dir, 'static', 'assets', 'confusion_matrix_category.png')
    plt.savefig(report_path)
    plt.savefig(asset_path)
    plt.close()
    print(f"Saved topic confusion matrix -> {report_path}")

    return {"accuracy": acc, "f1_macro": f1_macro, "classes": classes}


if __name__ == "__main__":
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    os.makedirs(os.path.join(root, 'reports'), exist_ok=True)
    os.makedirs(os.path.join(root, 'static', 'assets'), exist_ok=True)

    sent_metrics = evaluate_sentiment_model(root)
    cat_metrics = evaluate_category_model(root)

    print("\n================ BENCHMARK SUMMARY ================")
    print(f"  Sentiment Classifier Accuracy: {sent_metrics['accuracy'] * 100:.2f}%")
    print(f"  Topic Classifier Accuracy    : {cat_metrics['accuracy'] * 100:.2f}%")
    print("===================================================\n")

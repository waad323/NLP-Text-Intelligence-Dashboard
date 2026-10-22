"""Machine Learning Inference Engine for Sentiment Analysis and Topic Classification.

Provides unified, robust inference for:
  1. Sentiment Polarity: Binary Logistic Regression on IMDb movie reviews.
  2. News Topic Classification: 5-Class Logistic Regression on BBC News
     (Business, Entertainment, Politics, Sport, Tech).
Includes confidence scores and probability distributions.
"""

from dataclasses import dataclass
import logging
import os
import pickle
from typing import Any, Dict, Optional
import warnings

# Suppress sklearn unpickling warnings
warnings.filterwarnings('ignore', category=UserWarning)

from scripts.preprocess import clean_text

logger = logging.getLogger(__name__)


@dataclass
class SentimentResult:
    label: str
    confidence: float
    positive_prob: float
    negative_prob: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "label": self.label,
            "confidence": round(self.confidence * 100, 1),
            "positive_prob": round(self.positive_prob * 100, 1),
            "negative_prob": round(self.negative_prob * 100, 1),
        }


@dataclass
class CategoryResult:
    label: str
    confidence: float
    probabilities: Dict[str, float]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "label": self.label.capitalize(),
            "confidence": round(self.confidence * 100, 1),
            "probabilities": {k.capitalize(): round(v * 100, 1) for k, v in self.probabilities.items()},
        }


class MLPredictor:
    """Thread-safe inference manager for sentiment and topic classifiers."""

    _instance: Optional['MLPredictor'] = None

    def __init__(self, models_dir: Optional[str] = None):
        if models_dir is None:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            models_dir = os.path.abspath(os.path.join(base_dir, '..', 'saved_models'))

        self.models_dir = models_dir
        self.sentiment_model = None
        self.sentiment_tfidf = None
        self.category_model = None
        self.category_tfidf = None
        self._load_models()

    def _load_models(self) -> None:
        """Loads and prepares trained model artifacts from disk."""
        # 1. Sentiment Model Pipeline
        sent_model_path = os.path.join(self.models_dir, 'sentiment_model.pkl')
        sent_tfidf_path = os.path.join(self.models_dir, 'sentiment_tfidf.pkl')

        if os.path.exists(sent_model_path) and os.path.exists(sent_tfidf_path):
            try:
                with open(sent_model_path, 'rb') as f:
                    self.sentiment_model = pickle.load(f)
                with open(sent_tfidf_path, 'rb') as f:
                    self.sentiment_tfidf = pickle.load(f)

                # Ensure cross-version scikit-learn compatibility
                if not hasattr(self.sentiment_model, 'multi_class'):
                    self.sentiment_model.multi_class = 'auto'
            except Exception as e:
                logger.warning(f"Could not load sentiment model: {e}")

        # 2. Topic Classifier Pipeline
        cat_path = os.path.join(self.models_dir, 'text_classifier.pkl')
        if os.path.exists(cat_path):
            try:
                with open(cat_path, 'rb') as f:
                    data = pickle.load(f)
                    self.category_model = data['model']
                    self.category_tfidf = data['vectorizer']

                if not hasattr(self.category_model, 'multi_class'):
                    self.category_model.multi_class = 'auto'
            except Exception as e:
                logger.warning(f"Could not load topic classifier: {e}")

    def predict_sentiment(self, text: str) -> SentimentResult:
        """Predicts sentiment polarity (Positive / Negative) with confidence score."""
        if not self.sentiment_model or not self.sentiment_tfidf:
            return SentimentResult(label="Neutral", confidence=0.5, positive_prob=0.5, negative_prob=0.5)

        cleaned = clean_text(text)
        if not cleaned:
            return SentimentResult(label="Neutral", confidence=0.5, positive_prob=0.5, negative_prob=0.5)

        vector = self.sentiment_tfidf.transform([cleaned])
        probs = self.sentiment_model.predict_proba(vector)[0]
        classes = list(self.sentiment_model.classes_)

        pos_idx = classes.index('positive') if 'positive' in classes else 1
        neg_idx = classes.index('negative') if 'negative' in classes else 0

        pos_prob = float(probs[pos_idx])
        neg_prob = float(probs[neg_idx])

        if pos_prob >= neg_prob:
            return SentimentResult(label="Positive", confidence=pos_prob, positive_prob=pos_prob, negative_prob=neg_prob)
        else:
            return SentimentResult(label="Negative", confidence=neg_prob, positive_prob=pos_prob, negative_prob=neg_prob)

    def predict_category(self, text: str) -> CategoryResult:
        """Classifies text into BBC News topics with confidence score and class distribution."""
        if not self.category_model or not self.category_tfidf:
            return CategoryResult(label="General", confidence=0.0, probabilities={})

        cleaned = clean_text(text)
        if not cleaned:
            return CategoryResult(label="General", confidence=0.0, probabilities={})

        vector = self.category_tfidf.transform([cleaned])
        probs = self.category_model.predict_proba(vector)[0]
        classes = list(self.category_model.classes_)

        prob_dict = {classes[i]: float(probs[i]) for i in range(len(classes))}
        best_class = max(prob_dict, key=prob_dict.get)
        confidence = prob_dict[best_class]

        return CategoryResult(label=best_class, confidence=confidence, probabilities=prob_dict)


# Global singleton instance
_predictor = None


def get_predictor() -> MLPredictor:
    global _predictor
    if _predictor is None:
        _predictor = MLPredictor()
    return _predictor


def predict_sentiment(text: str) -> str:
    """Convenience string helper returning 'Positive' or 'Negative'."""
    return get_predictor().predict_sentiment(text).label


def predict_category(text: str) -> str:
    """Convenience string helper returning predicted news category."""
    return get_predictor().predict_category(text).label.capitalize()


if __name__ == "__main__":
    sample = "Alphabet reported a 28% surge in operating profit driven by Google Cloud and digital ad growth."
    pred = get_predictor()
    print("Sentiment:", pred.predict_sentiment(sample).to_dict())
    print("Category :", pred.predict_category(sample).to_dict())

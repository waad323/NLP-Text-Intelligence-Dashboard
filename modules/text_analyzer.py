"""Custom Extractive Text Summarization and Keyword Extraction Engine.

Implements mathematical sentence ranking and keyphrase identification
using TF-IDF vectorization without relying on black-box external summarizers.
"""

import math
import re
from typing import Dict, List, Tuple
import nltk
from nltk.tokenize import sent_tokenize
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

# Ensure punkt sentence tokenizer is available
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab', quiet=True)


def clean_sentence_for_tfidf(sentence: str) -> str:
    """Normalizes a sentence for TF-IDF feature extraction."""
    if not isinstance(sentence, str):
        return ""
    sentence = sentence.lower().replace("\n", " ")
    sentence = re.sub(r'https?://\S+|www\.\S+', '', sentence)
    sentence = re.sub(r'[^a-zA-Z0-9\s]', ' ', sentence)
    sentence = re.sub(r'\s+', ' ', sentence)
    return sentence.strip()


def compute_text_stats(text: str) -> Dict[str, float]:
    """Computes basic quantitative linguistic statistics for the input text."""
    words = re.findall(r'\b[a-zA-Z]{2,}\b', text.lower())
    sentences = sent_tokenize(text.strip()) if text.strip() else []

    word_count = len(words)
    sentence_count = len(sentences)
    unique_words = len(set(words))
    lexical_diversity = round((unique_words / word_count) * 100, 1) if word_count > 0 else 0.0

    # Average human reading speed: ~200-220 words per minute
    reading_seconds = max(1, math.ceil((word_count / 200) * 60)) if word_count > 0 else 0

    return {
        "word_count": word_count,
        "sentence_count": sentence_count,
        "unique_words": unique_words,
        "lexical_diversity": lexical_diversity,
        "reading_time_sec": reading_seconds,
    }


def analyze_text(text: str, num_sentences: int = 3, num_keywords: int = 6) -> Tuple[str, List[str], Dict[str, float]]:
    """Performs extractive summarization and TF-IDF keyword extraction.

    Algorithm:
      1. Sentence Segmentation: Breaks text into candidate sentences.
      2. Vectorization: Fits a Unigram + Bigram TF-IDF matrix across sentences.
      3. Scoring: Calculates vector norms / summation per sentence to gauge information density.
      4. Extractive Selection: Ranks top N sentences and restores original chronological sequence.
      5. Keyword Extraction: Computes document-level TF-IDF weights and selects top N keyphrases.
      6. Text Statistics: Computes reading metrics (word count, sentence count, reading speed).

    Args:
        text: Input raw text string.
        num_sentences: Target summary sentence count (default: 3).
        num_keywords: Target number of keywords to extract (default: 6).

    Returns:
        Tuple of (summary_text, keywords_list, stats_dict)
    """
    if not text or not text.strip():
        return "No text provided to analyze.", [], compute_text_stats("")

    stats = compute_text_stats(text)
    raw_sentences = [s.strip() for s in sent_tokenize(text.strip()) if s.strip()]

    # Edge cases: Short texts where sentence count <= requested summary
    if len(raw_sentences) <= num_sentences:
        summary = " ".join(raw_sentences)
    else:
        cleaned_sentences = [clean_sentence_for_tfidf(s) for s in raw_sentences]
        valid_sentences = [s for s in cleaned_sentences if len(s.split()) >= 3]

        if not valid_sentences or len(valid_sentences) < num_sentences:
            summary = " ".join(raw_sentences[:num_sentences])
        else:
            try:
                vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
                tfidf_matrix = vectorizer.fit_transform(cleaned_sentences)

                # Sentence scores based on total TF-IDF information weight
                scores = np.array(tfidf_matrix.sum(axis=1)).ravel()

                # Pick top scoring indices and preserve original chronological order
                ranked_indices = np.argsort(scores)[::-1]
                selected_indices = sorted(ranked_indices[:num_sentences])
                summary = " ".join([raw_sentences[i] for i in selected_indices])
            except (ValueError, IndexError):
                summary = " ".join(raw_sentences[:num_sentences])

    # Extract Top N Keywords using document-level TF-IDF
    keywords: List[str] = []
    try:
        full_cleaned = clean_sentence_for_tfidf(text)
        if full_cleaned:
            kw_vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2), max_features=100)
            kw_matrix = kw_vectorizer.fit_transform([full_cleaned])
            feature_names = np.array(kw_vectorizer.get_feature_names_out())
            kw_scores = np.array(kw_matrix.sum(axis=0)).ravel()

            top_kw_indices = np.argsort(kw_scores)[::-1][:num_keywords]
            keywords = [feature_names[i] for i in top_kw_indices if len(feature_names[i]) > 2]
    except (ValueError, IndexError):
        keywords = []

    return summary, keywords, stats


if __name__ == "__main__":
    sample = (
        "Artificial Intelligence is transforming enterprise operations across finance, healthcare, and engineering. "
        "Deep learning architectures enable automated pattern discovery from complex historical data repositories. "
        "Natural language processing pipelines extract structural insights, sentiment polarities, and named entities. "
        "Building custom algorithmic engines gives developers complete transparency over statistical decision boundaries. "
        "Modern cloud deployments package these analytical components into high-throughput production microservices."
    )
    s, kw, st = analyze_text(sample, num_sentences=2, num_keywords=4)
    print("Summary:", s)
    print("Keywords:", kw)
    print("Stats:", st)

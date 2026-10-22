"""Text Preprocessing and Normalization Pipeline.

Provides reusable text cleaning, regex normalization, stopword removal,
and lemmatization for NLP modeling and inference.
"""

import re
from typing import Optional
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Ensure essential NLTK lexical resources are downloaded quietly
for resource in ['corpora/stopwords', 'corpora/wordnet', 'corpora/omw-1.4']:
    try:
        nltk.data.find(resource)
    except LookupError:
        res_name = resource.split('/')[-1]
        nltk.download(res_name, quiet=True)

_STOPWORDS = set(stopwords.words('english'))
_LEMMATIZER = WordNetLemmatizer()


def clean_text(text: Optional[str], remove_stopwords: bool = True, lemmatize: bool = True) -> str:
    """Cleans and normalizes raw text for feature extraction.

    Transformations applied:
      1. Strips HTML tags (e.g., <br />, <div>).
      2. Converts to lowercase.
      3. Strips URLs and hyper-references.
      4. Strips non-alphanumeric punctuation and standalone numbers.
      5. Normalizes whitespace sequences.
      6. Filters out standard English stopwords (optional).
      7. Applies WordNet morphological lemmatization (optional).

    Args:
        text: Raw text string.
        remove_stopwords: If True, filters out standard English stopwords.
        lemmatize: If True, applies WordNet lemmatization.

    Returns:
        Cleaned, normalized string ready for TF-IDF vectorization.
    """
    if not text or not isinstance(text, str):
        return ""

    # Strip HTML tags
    cleaned = re.sub(r'<[^>]+>', ' ', text)

    # Lowercase
    cleaned = cleaned.lower()

    # Strip URLs
    cleaned = re.sub(r'https?://\S+|www\.\S+', ' ', cleaned)

    # Remove email addresses
    cleaned = re.sub(r'\S+@\S+', ' ', cleaned)

    # Remove non-alphabet characters (retain spaces)
    cleaned = re.sub(r'[^a-zA-Z\s]', ' ', cleaned)

    # Normalize multiple whitespace characters
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()

    if not cleaned:
        return ""

    words = cleaned.split()

    if remove_stopwords:
        words = [w for w in words if w not in _STOPWORDS and len(w) > 1]

    if lemmatize:
        words = [_LEMMATIZER.lemmatize(w) for w in words]

    return " ".join(words)


if __name__ == "__main__":
    sample = "<p>Apple Inc. broke records today! Check https://apple.com -- running & testing models 100%.</p>"
    print("Original:", sample)
    print("Cleaned :", clean_text(sample))

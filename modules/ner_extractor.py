"""Named Entity Recognition (NER) Module.

Extracts entities (People, Organizations, Locations, Dates, Currencies)
using spaCy pre-trained models with frequency tracking and structured categorization.
"""

from collections import Counter
import logging
from typing import Dict, List, Set
import spacy

logger = logging.getLogger(__name__)

# Load spaCy English pipeline
try:
    _nlp = spacy.load("en_core_web_sm")
except Exception:
    try:
        from spacy.cli import download
        download("en_core_web_sm")
        _nlp = spacy.load("en_core_web_sm")
    except Exception as e:
        logger.warning(f"Could not load spaCy en_core_web_sm model: {e}")
        _nlp = None

# Entity label mapping for user-friendly UI presentation
_LABEL_MAPPING = {
    "PERSON": "Person",
    "ORG": "Organization",
    "GPE": "Location",
    "LOC": "Location",
    "DATE": "Date / Time",
    "TIME": "Date / Time",
    "MONEY": "Financial",
    "PRODUCT": "Product",
    "EVENT": "Event",
    "FAC": "Facility",
    "NORP": "Nationality / Group",
}


def extract_entities(text: str) -> Dict[str, List[str]]:
    """Extracts named entities from input text grouped by high-level category.

    Args:
        text: Raw text string.

    Returns:
        Dictionary mapping category names (e.g., 'Person', 'Organization')
        to alphabetically sorted unique entity names.
    """
    if not text or not text.strip():
        return {
            "Person": [],
            "Organization": [],
            "Location": [],
            "Financial": [],
            "Date / Time": [],
        }

    if _nlp is None:
        return {"Person": [], "Organization": [], "Location": []}

    doc = _nlp(text[:50000])  # Protect against memory blowup on massive text
    categorized: Dict[str, Set[str]] = {
        "Person": set(),
        "Organization": set(),
        "Location": set(),
        "Financial": set(),
        "Date / Time": set(),
    }

    for ent in doc.ents:
        mapped_cat = _LABEL_MAPPING.get(ent.label_)
        if mapped_cat:
            clean_ent = ent.text.strip().replace("\n", " ")
            if len(clean_ent) > 1 and not clean_ent.isnumeric():
                categorized.setdefault(mapped_cat, set()).add(clean_ent)

    # Convert sets to sorted lists, filtering out empty categories
    return {
        cat: sorted(list(entities))
        for cat, entities in categorized.items()
        if entities
    }


def extract_entities_detailed(text: str) -> List[Dict[str, str]]:
    """Extracts entity mentions with their start/end offsets and frequencies for highlight rendering."""
    if not text or not text.strip() or _nlp is None:
        return []

    doc = _nlp(text[:50000])
    results = []
    for ent in doc.ents:
        mapped_cat = _LABEL_MAPPING.get(ent.label_, ent.label_)
        results.append({
            "text": ent.text.strip(),
            "label": mapped_cat,
            "start": ent.start_char,
            "end": ent.end_char,
        })
    return results


if __name__ == "__main__":
    sample = (
        "Tim Cook announced in Cupertino, California that Apple Inc. generated $90 billion "
        "in revenue in January 2026. The Wall Street Journal reported on the European summit in Brussels."
    )
    print("Extracted Entities:", extract_entities(sample))

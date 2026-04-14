"""
Text preprocessing utilities shared across weeks.
Handles stemming, stopword removal, and domain-specific
normalization of numbers and units (e.g. "8 ft.", "mdf 3/4").
"""

import re
import nltk
from nltk.stem.porter import PorterStemmer

nltk.download("stopwords", quiet=True)
from nltk.corpus import stopwords

STEMMER = PorterStemmer()
STOP_WORDS = set(stopwords.words("english"))

# Unit aliases for domain normalization
UNIT_MAP = {
    r"\bft\.?\b": "foot",
    r"\bin\.?\b": "inch",
    r"\blbs?\.?\b": "pound",
    r"\boz\.?\b": "ounce",
    r"\bsq\.?\b": "square",
    r"\bpk\.?\b": "pack",
}


def normalize_units(text: str) -> str:
    """Replace common unit abbreviations with canonical forms."""
    text = text.lower()
    for pattern, replacement in UNIT_MAP.items():
        text = re.sub(pattern, replacement, text)
    return text


def tokenize(text: str) -> list[str]:
    """Lowercase, strip punctuation, split on whitespace."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return text.split()


def stem_tokens(tokens: list[str]) -> list[str]:
    return [STEMMER.stem(t) for t in tokens]


def remove_stopwords(tokens: list[str]) -> list[str]:
    return [t for t in tokens if t not in STOP_WORDS]


def preprocess(text: str, stem: bool = True) -> list[str]:
    """Full pipeline: normalize units → tokenize → remove stopwords → (stem)."""
    text = normalize_units(str(text))
    tokens = tokenize(text)
    tokens = remove_stopwords(tokens)
    if stem:
        tokens = stem_tokens(tokens)
    return tokens

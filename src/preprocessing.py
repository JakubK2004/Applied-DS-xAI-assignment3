"""
Text preprocessing utilities shared across weeks.
Handles data loading, stemming, stopword removal, and domain-specific
normalization of numbers and units (e.g. "8 ft.", "mdf 3/4").
"""

import re
import pandas as pd
import nltk
from nltk.stem.porter import PorterStemmer
from pathlib import Path

nltk.download("stopwords", quiet=True)
from nltk.corpus import stopwords

DATA_DIR = Path(__file__).parent.parent / "data"

# ---------------------------------------------------------------------------
# Data loading & preparation
# ---------------------------------------------------------------------------

def prepare_data(stem: bool = True):
    """
    Load all CSVs, clean text columns, and return ready-to-use DataFrames.

    Parameters
    ----------
    stem : bool
        Whether to apply stemming during text preprocessing (default True).
        Set to False to compare results without stemming (Week 10 task).

    Returns
    -------
    train       : DataFrame — training set with cleaned text columns
    test        : DataFrame — test set with cleaned text columns
    attributes  : DataFrame — raw attributes (all products)
    attr_lookup : dict      — product_uid -> concatenated attribute string
    """
    train        = pd.read_csv(DATA_DIR / "train.csv", encoding="ISO-8859-1")
    test         = pd.read_csv(DATA_DIR / "test.csv", encoding="ISO-8859-1")
    descriptions = pd.read_csv(DATA_DIR / "product_descriptions.csv", encoding="ISO-8859-1")
    attributes   = pd.read_csv(DATA_DIR / "attributes.csv", encoding="ISO-8859-1")

    # Merge product descriptions into train and test
    train = train.merge(descriptions, on="product_uid", how="left")
    test  = test.merge(descriptions, on="product_uid", how="left")

    # Fill missing text fields with empty string
    text_cols = ["search_term", "product_title", "product_description"]
    for col in text_cols:
        train[col] = train[col].fillna("")
        test[col]  = test[col].fillna("")

    # Apply text preprocessing to all text columns
    for col in text_cols:
        train[col] = train[col].map(lambda t: " ".join(preprocess(str(t), stem=stem)))
        test[col]  = test[col].map(lambda t: " ".join(preprocess(str(t), stem=stem)))

    # Build attribute lookup: product_uid -> concatenated attribute string
    attr_lookup = (
        attributes.dropna(subset=["value"])
        .groupby("product_uid")["value"]
        .apply(lambda vals: " ".join(str(v) for v in vals))
        .to_dict()
    )

    return train, test, attributes, attr_lookup

# ---------------------------------------------------------------------------
# Text preprocessing
# ---------------------------------------------------------------------------

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
    """Full pipeline: normalize units -> tokenize -> remove stopwords -> (stem)."""
    text = normalize_units(str(text))
    tokens = tokenize(text)
    tokens = remove_stopwords(tokens)
    if stem:
        tokens = stem_tokens(tokens)
    return tokens

"""
Feature extraction for query-product matching.
All feature functions take preprocessed token lists or raw strings
and return a numeric value used as a model input.
"""

import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from preprocessing import preprocess


# ---------------------------------------------------------------------------
# Term-count overlap (baseline, from Yao-Jen Chang)
# ---------------------------------------------------------------------------

def count_common_words(query_tokens: list[str], target_tokens: list[str]) -> int:
    """Number of query tokens that appear in target tokens."""
    target_set = set(target_tokens)
    return sum(1 for t in query_tokens if t in target_set)


# ---------------------------------------------------------------------------
# TF-IDF cosine similarity
# ---------------------------------------------------------------------------

def tfidf_similarity(query: str, document: str, vectorizer: TfidfVectorizer) -> float:
    """Cosine similarity between query and document using a fitted TF-IDF vectorizer."""
    vecs = vectorizer.transform([query, document])
    return float(cosine_similarity(vecs[0], vecs[1])[0, 0])


# ---------------------------------------------------------------------------
# Attribute-based features
# ---------------------------------------------------------------------------

def query_in_attributes(query_tokens: list[str], attr_text: str) -> int:
    """Count of query tokens found in the product attributes string."""
    attr_tokens = set(preprocess(attr_text))
    return sum(1 for t in query_tokens if t in attr_tokens)


# ---------------------------------------------------------------------------
# Domain-specific: number and unit matching
# ---------------------------------------------------------------------------

_NUMBER_PATTERN = re.compile(r"\d+(?:[./]\d+)?")


def extract_numbers(text: str) -> set[str]:
    return set(_NUMBER_PATTERN.findall(text))


def number_match_score(query: str, target: str) -> float:
    """Fraction of numeric tokens in the query that also appear in target."""
    q_nums = extract_numbers(query)
    if not q_nums:
        return 0.0
    t_nums = extract_numbers(target)
    return len(q_nums & t_nums) / len(q_nums)


# ---------------------------------------------------------------------------
# Feature vector builder
# ---------------------------------------------------------------------------

def build_feature_vector(
    row,
    attr_lookup: dict,
    tfidf_vec: TfidfVectorizer | None = None,
    stem: bool = True,
) -> list[float]:
    """
    Build a feature vector for a single (query, product) row.

    Parameters
    ----------
    row : pandas Series with fields: search_term, product_title, product_description
    attr_lookup : dict mapping product_uid -> concatenated attribute string
    tfidf_vec : fitted TfidfVectorizer (optional; skip TF-IDF features if None)
    stem : whether to apply stemming during preprocessing
    """
    query = str(row["search_term"])
    title = str(row["product_title"])
    desc = str(row["product_description"])
    uid = row["product_uid"]
    attrs = attr_lookup.get(uid, "")

    q_tokens = preprocess(query, stem=stem)
    t_tokens = preprocess(title, stem=stem)
    d_tokens = preprocess(desc, stem=stem)

    features = [
        count_common_words(q_tokens, t_tokens),          # overlap: query vs title
        count_common_words(q_tokens, d_tokens),          # overlap: query vs description
        query_in_attributes(q_tokens, attrs),            # overlap: query vs attributes
        number_match_score(query, title),                # numeric match: title
        number_match_score(query, desc),                 # numeric match: description
        len(q_tokens),                                   # query length
    ]

    if tfidf_vec is not None:
        features.append(tfidf_similarity(query, title, tfidf_vec))
        features.append(tfidf_similarity(query, desc, tfidf_vec))

    return features


FEATURE_NAMES_BASE = [
    "overlap_title",
    "overlap_desc",
    "overlap_attrs",
    "num_match_title",
    "num_match_desc",
    "query_len",
]

FEATURE_NAMES_TFIDF = FEATURE_NAMES_BASE + ["tfidf_title", "tfidf_desc"]

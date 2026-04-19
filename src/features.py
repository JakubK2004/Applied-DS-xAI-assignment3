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
# Ratio features
# ---------------------------------------------------------------------------

def overlap_ratio(overlap: int, length: int) -> float:
    """Overlap count normalised by a length (avoids division by zero)."""
    return overlap / length if length > 0 else 0.0


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


def brand_match(query_tokens: list[str], brand: str) -> int:
    """1 if any brand token appears in the query."""
    if not brand:
        return 0
    brand_tokens = set(preprocess(brand))
    return int(any(t in brand_tokens for t in query_tokens))


def query_is_brand(query_tokens: list[str], brand: str) -> int:
    """1 if the query tokens exactly match the brand tokens."""
    if not brand or not query_tokens:
        return 0
    brand_tokens = set(preprocess(brand))
    return int(set(query_tokens) == brand_tokens)


# ---------------------------------------------------------------------------
# Query term position in title
# ---------------------------------------------------------------------------

def query_term_position_score(query_tokens: list[str], title_tokens: list[str]) -> float:
    """
    Average normalised position of matched query terms in the title.
    Returns 1.0 if all matches are at the start, 0.0 if none match or all at the end.
    Earlier matches = stronger relevance signal.
    """
    if not title_tokens:
        return 0.0
    title_len = len(title_tokens)
    positions = [
        i / title_len
        for t in query_tokens
        for i, tok in enumerate(title_tokens)
        if tok == t
    ]
    if not positions:
        return 0.0
    return 1.0 - (sum(positions) / len(positions))


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


def query_has_number(query: str) -> int:
    """1 if the query contains any numeric token."""
    return int(bool(extract_numbers(query)))


# ---------------------------------------------------------------------------
# spaCy semantic similarity
# ---------------------------------------------------------------------------

def spacy_similarity(query: str, document: str, nlp) -> float:
    """Semantic similarity between query and document using spaCy word vectors."""
    doc_q = nlp(query)
    doc_d = nlp(document)
    if not doc_q.has_vector or not doc_d.has_vector:
        return 0.0
    return float(doc_q.similarity(doc_d))


# ---------------------------------------------------------------------------
# Feature vector builder
# ---------------------------------------------------------------------------

def build_feature_vector(
    row,
    attr_lookup: dict,
    tfidf_vec: TfidfVectorizer | None = None,
    brand_lookup: dict | None = None,
    num_attrs_lookup: dict | None = None,
    nlp=None,
    stem: bool = True,
) -> list[float]:
    """
    Build a feature vector for a single (query, product) row.

    Parameters
    ----------
    row            : pandas Series with fields: search_term, product_title, product_description
    attr_lookup    : product_uid -> concatenated attribute string
    tfidf_vec      : fitted TfidfVectorizer (optional)
    brand_lookup   : product_uid -> brand name string (optional)
    num_attrs_lookup: product_uid -> attribute count (optional)
    nlp            : spaCy language model (optional)
    stem           : whether to apply stemming during preprocessing
    """
    query = str(row["search_term"])
    title = str(row["product_title"])
    desc = str(row["product_description"])
    uid = row["product_uid"]
    attrs = attr_lookup.get(uid, "")
    brand = brand_lookup.get(uid, "") if brand_lookup else ""

    q_tokens = preprocess(query, stem=stem)
    t_tokens = preprocess(title, stem=stem)
    d_tokens = preprocess(desc, stem=stem)

    ov_title = count_common_words(q_tokens, t_tokens)
    ov_desc  = count_common_words(q_tokens, d_tokens)
    q_len    = len(q_tokens)

    features = [
        ov_title,                                        # overlap: query vs title
        ov_desc,                                         # overlap: query vs description
        query_in_attributes(q_tokens, attrs),            # overlap: query vs attributes
        number_match_score(query, title),                # numeric match: title
        number_match_score(query, desc),                 # numeric match: description
        q_len,                                           # query length
        overlap_ratio(ov_title, q_len),                  # fraction of query matched in title
        overlap_ratio(ov_desc, q_len),                   # fraction of query matched in desc
        overlap_ratio(ov_title, len(t_tokens)),          # fraction of title covered by query
        overlap_ratio(ov_desc, len(d_tokens)),           # fraction of desc covered by query
        brand_match(q_tokens, brand),                    # query contains brand name
        query_is_brand(q_tokens, brand),                 # query is exactly the brand
        num_attrs_lookup.get(uid, 0) if num_attrs_lookup else 0,  # number of attributes
        query_has_number(query),                         # query contains a number
        query_term_position_score(q_tokens, t_tokens),  # avg position of matches in title
    ]

    if tfidf_vec is not None:
        features.append(tfidf_similarity(query, title, tfidf_vec))
        features.append(tfidf_similarity(query, desc, tfidf_vec))

    if nlp is not None:
        features.append(spacy_similarity(query, title, nlp))
        features.append(spacy_similarity(query, desc, nlp))

    return features


FEATURE_NAMES_BASE = [
    "overlap_title",
    "overlap_desc",
    "overlap_attrs",
    "num_match_title",
    "num_match_desc",
    "query_len",
    "overlap_title_ratio",
    "overlap_desc_ratio",
    "title_coverage",
    "desc_coverage",
    "brand_match",
    "query_is_brand",
    "num_attributes",
    "query_has_number",
    "query_term_position",
]

FEATURE_NAMES_TFIDF = FEATURE_NAMES_BASE + ["tfidf_title", "tfidf_desc"]

FEATURE_NAMES_FULL = FEATURE_NAMES_TFIDF + ["spacy_title", "spacy_desc"]

# Train BaggingRF on incrementally richer feature sets and compare RMSE.

import sys
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

sys.path.insert(0, ".")
from features import build_feature_vector, FEATURE_NAMES_TFIDF

DATA_DIR = "data"
RANDOM_STATE = 42

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

train        = pd.read_csv(f"{DATA_DIR}/train.csv", encoding="ISO-8859-1")
descriptions = pd.read_csv(f"{DATA_DIR}/product_descriptions.csv", encoding="ISO-8859-1")
attributes   = pd.read_csv(f"{DATA_DIR}/attributes.csv", encoding="ISO-8859-1")

train = train.merge(descriptions, on="product_uid", how="left")

attr_lookup = (
    attributes.dropna(subset=["value"])
    .groupby("product_uid")["value"]
    .apply(lambda v: " ".join(v.astype(str)))
    .to_dict()
)

# ---------------------------------------------------------------------------
# Fit TF-IDF on title + description corpus (training set only)
# ---------------------------------------------------------------------------

corpus = pd.concat([train["product_title"], train["product_description"]]).astype(str)
tfidf = TfidfVectorizer(max_features=5000)
tfidf.fit(corpus)

# ---------------------------------------------------------------------------
# Build full 8-feature matrix once
# ---------------------------------------------------------------------------

print("Building feature matrix (this takes a few minutes)...")
X_full = np.array([
    build_feature_vector(row, attr_lookup, tfidf_vec=tfidf, stem=True)
    for _, row in train.iterrows()
])
y = train["relevance"].values
print(f"Feature matrix shape: {X_full.shape}")

# ---------------------------------------------------------------------------
# Train/test split — same seed as baseline for comparability
# ---------------------------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X_full, y, test_size=0.2, random_state=RANDOM_STATE
)

# ---------------------------------------------------------------------------
# Feature set definitions (column index subsets)
# ---------------------------------------------------------------------------
# Column order (FEATURE_NAMES_TFIDF):
#   0: overlap_title   1: overlap_desc   2: overlap_attrs
#   3: num_match_title 4: num_match_desc  5: query_len
#   6: tfidf_title     7: tfidf_desc

FEATURE_SETS = [
    ("Baseline (Yao-Jen Chang)",  [0, 1, 5]),
    ("+ Attribute overlap",       [0, 1, 5, 2]),
    ("+ Number matching",         [0, 1, 5, 2, 3, 4]),
    ("+ TF-IDF similarity",       [0, 1, 5, 2, 3, 4, 6, 7]),
]

# Same config as the Yao-Jen Chang baseline

def make_model():
    rf = RandomForestRegressor(
        n_estimators=15, max_depth=6, random_state=0, n_jobs=-1
    )
    return BaggingRegressor(
        estimator=rf, n_estimators=40, max_samples=0.1,
        random_state=RANDOM_STATE, n_jobs=-1,
    )

# ---------------------------------------------------------------------------
# Run comparison
# ---------------------------------------------------------------------------

print("\n{:<30s}  {:>6s}  {:>5s}".format("Feature set", "RMSE", "#feat"))
print("-" * 48)

for name, cols in FEATURE_SETS:
    Xtr = X_train[:, cols]
    Xte = X_test[:, cols]

    model = make_model()
    model.fit(Xtr, y_train)
    rmse = mean_squared_error(y_test, model.predict(Xte)) ** 0.5

    print(f"{name:<30s}  {rmse:.4f}  {len(cols):>5d}")

print("\nDone. Lower RMSE = better.")

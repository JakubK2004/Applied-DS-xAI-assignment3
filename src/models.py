# Compare regression models on RMSE and training time, then tune the best one.

import sys
import time
import hashlib
import pickle
import numpy as np
import pandas as pd
import spacy
from xgboost import XGBRegressor
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV
from sklearn.metrics import root_mean_squared_error
from sklearn.feature_extraction.text import TfidfVectorizer

sys.path.insert(0, str(Path(__file__).parent))
from preprocessing import preprocess
from features import build_feature_vector, FEATURE_NAMES_FULL  # noqa

DATA_DIR  = Path(__file__).parent.parent / "data"
CACHE_DIR = DATA_DIR / ".cache"
RANDOM_STATE = 42

# Load data

print("[1/6] Loading data...")
t0 = time.time()
train = pd.read_csv(DATA_DIR / "train.csv", encoding="ISO-8859-1")
descriptions = pd.read_csv(DATA_DIR / "product_descriptions.csv")
attributes = pd.read_csv(DATA_DIR / "attributes.csv")
train = train.merge(descriptions, on="product_uid", how="left")
print(f"      {len(train):,} training rows loaded  ({time.time()-t0:.1f}s)")

attrs_clean = attributes.dropna(subset=["value"])

attr_lookup = (
    attrs_clean
    .groupby("product_uid")["value"]
    .apply(lambda v: " ".join(v.astype(str)))
    .to_dict()
)

brand_lookup = (
    attrs_clean[attrs_clean["name"] == "MFG Brand Name"]
    .set_index("product_uid")["value"]
    .to_dict()
)

num_attrs_lookup = (
    attrs_clean.groupby("product_uid")["value"]
    .count()
    .to_dict()
)

# Feature matrix — build or load from cache

source_files = [DATA_DIR / "train.csv", DATA_DIR / "product_descriptions.csv", DATA_DIR / "attributes.csv"]
mtimes = "".join(f"{p.stat().st_mtime}" for p in source_files)
cache_key  = hashlib.md5(f"{mtimes}v1".encode()).hexdigest()
cache_file = CACHE_DIR / f"features_{cache_key}.pkl"

if cache_file.exists():
    print("[2/6] Loading feature matrix from cache...")
    t0 = time.time()
    with open(cache_file, "rb") as f:
        X, y = pickle.load(f)
    print(f"      Feature matrix: {X.shape}  ({time.time()-t0:.1f}s)")
else:
    print("[2/6] Loading spaCy model (en_core_web_md)...")
    t0 = time.time()
    nlp = spacy.load("en_core_web_md")
    print(f"      Done  ({time.time()-t0:.1f}s)")

    print("[3/6] Fitting TF-IDF vectorizer...")
    corpus = pd.concat([train["product_title"], train["product_description"]]).astype(str)
    tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    tfidf.fit(corpus)
    print(f"      Vocabulary size: {len(tfidf.vocabulary_):,}")

    print("[4/6] Pre-computing spaCy similarities in batch...")
    t0 = time.time()
    queries = train["search_term"].fillna("").tolist()
    titles  = train["product_title"].fillna("").tolist()
    descs   = train["product_description"].fillna("").tolist()

    query_docs = list(nlp.pipe(queries, batch_size=512))
    title_docs = list(nlp.pipe(titles,  batch_size=512))
    desc_docs  = list(nlp.pipe(descs,   batch_size=512))

    spacy_sims = [
        (
            float(q.similarity(t)) if q.has_vector and t.has_vector else 0.0,
            float(q.similarity(d)) if q.has_vector and d.has_vector else 0.0,
        )
        for q, t, d in zip(query_docs, title_docs, desc_docs)
    ]
    print(f"      spaCy done  ({time.time()-t0:.1f}s)")

    print("[5/6] Building feature matrix...")
    t0 = time.time()
    X = np.array([
        build_feature_vector(
            row, attr_lookup,
            tfidf_vec=tfidf,
            brand_lookup=brand_lookup,
            num_attrs_lookup=num_attrs_lookup,
            spacy_sims=sims,
            stem=True,
        )
        for (_, row), sims in zip(train.iterrows(), spacy_sims)
    ])
    y = train["relevance"].values
    print(f"      Feature matrix: {X.shape}  ({time.time()-t0:.1f}s)")

    print("      Saving to cache...")
    CACHE_DIR.mkdir(exist_ok=True)
    with open(cache_file, "wb") as f:
        pickle.dump((X, y), f)
    print("      Cached.")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE
)
print(f"      Train: {len(X_train):,} rows  |  Test: {len(X_test):,} rows")

# Model comparison

print("\n[6/6] Training, evaluating, and tuning models...")
MODELS = {
    "BaggingRF (baseline)": BaggingRegressor(
        estimator=RandomForestRegressor(n_estimators=15, max_depth=6, random_state=0, n_jobs=-1),
        n_estimators=40, max_samples=0.1, random_state=42, n_jobs=-1,
    ),
    "GradientBoosting": GradientBoostingRegressor(n_estimators=100, random_state=RANDOM_STATE),
    "XGBoost": XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=RANDOM_STATE, n_jobs=-1, verbosity=0),
    "Ridge": Ridge(),
    "KNeighbors": KNeighborsRegressor(n_neighbors=10, n_jobs=-1),
}

results = {}
for name, model in MODELS.items():
    print(f"      Training {name}...")
    t0 = time.time()
    model.fit(X_train, y_train)
    elapsed = time.time() - t0
    rmse = root_mean_squared_error(y_test, model.predict(X_test))
    results[name] = {"RMSE": rmse, "Train time (s)": round(elapsed, 2)}
    print(f"      {name:30s}  RMSE={rmse:.4f}  time={elapsed:.1f}s")

# Hyperparameter tuning

best_name = min(results, key=lambda k: results[k]["RMSE"])
print(f"\n  Hyperparameter tuning — best so far: {best_name}")

# Stage 1: broad random search

print("\n  Stage 1: RandomizedSearchCV (wide grid, 40 iterations)...")
broad_param_dist = {
    "n_estimators":      [50, 100, 200, 300, 500],
    "max_depth":         [2, 3, 4, 5, 6, 7, 8],
    "learning_rate":     [0.01, 0.05, 0.1, 0.15, 0.2, 0.3],
    "subsample":         [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    "colsample_bytree":  [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    "reg_alpha":         [0, 0.01, 0.1, 0.5, 1.0],
    "reg_lambda":        [0.5, 1.0, 2.0, 5.0],
    "min_child_weight":  [1, 3, 5, 10],
}

search = RandomizedSearchCV(
    XGBRegressor(random_state=RANDOM_STATE, n_jobs=-1, verbosity=0),
    param_distributions=broad_param_dist,
    n_iter=40,
    cv=5,
    scoring="neg_root_mean_squared_error",
    random_state=RANDOM_STATE,
    n_jobs=-1,
    verbose=1,
)
search.fit(X_train, y_train)
p = search.best_params_
rmse_stage1 = root_mean_squared_error(y_test, search.best_estimator_.predict(X_test))
print(f"\n  Stage 1 best params: {p}")
print(f"  Stage 1 RMSE:        {rmse_stage1:.4f}")

# Stage 2: narrow grid around Stage 1 best

print("\n  Stage 2: GridSearchCV (narrow grid around Stage 1 best)...")

def neighbours(val, candidates):
    """Return candidates within one step of val."""
    idx = candidates.index(val) if val in candidates else 0
    return candidates[max(0, idx-1): idx+2]

n_est_grid  = sorted(set([max(50, p["n_estimators"] - 50), p["n_estimators"], p["n_estimators"] + 50]))
depth_grid  = sorted(set([max(2, p["max_depth"] - 1), p["max_depth"], p["max_depth"] + 1]))
lr_candidates = [0.01, 0.05, 0.1, 0.15, 0.2, 0.3]
lr_grid     = neighbours(p["learning_rate"], lr_candidates)
sub_candidates = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
sub_grid    = neighbours(p["subsample"], sub_candidates)
col_candidates = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
col_grid    = neighbours(p["colsample_bytree"], col_candidates)

narrow_grid = {
    "n_estimators":     n_est_grid,
    "max_depth":        depth_grid,
    "learning_rate":    lr_grid,
    "subsample":        sub_grid,
    "colsample_bytree": col_grid,
}
print(f"  Narrow grid: {narrow_grid}")

fine_search = GridSearchCV(
    XGBRegressor(
        reg_alpha=p["reg_alpha"],
        reg_lambda=p["reg_lambda"],
        min_child_weight=p["min_child_weight"],
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbosity=0,
    ),
    param_grid=narrow_grid,
    cv=5,
    scoring="neg_root_mean_squared_error",
    n_jobs=-1,
    verbose=1,
)
fine_search.fit(X_train, y_train)
rmse_stage2 = root_mean_squared_error(y_test, fine_search.best_estimator_.predict(X_test))
print(f"\n  Stage 2 best params: {fine_search.best_params_}")
print(f"  Stage 2 RMSE:        {rmse_stage2:.4f}")
print(f"\n  Improvement over untuned XGBoost: {results['XGBoost']['RMSE'] - rmse_stage2:+.4f}")

# expose best estimator for feature_analysis.py
search = fine_search

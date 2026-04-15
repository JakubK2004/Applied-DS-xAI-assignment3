"""
Model comparison and hyperparameter optimization — Week 11 tasks.

Tasks:
1. Compare 4 regression models (RMSE + training time)
2. Hyperparameter optimization with RandomizedSearchCV on the best model
3. Evaluate effect of preprocessing and domain-specific features
"""

import sys
import time
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.feature_extraction.text import TfidfVectorizer

sys.path.insert(0, ".")
from preprocessing import preprocess
from features import build_feature_vector, FEATURE_NAMES_TFIDF

DATA_DIR = "data"
RANDOM_STATE = 42

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

train = pd.read_csv(f"{DATA_DIR}/train.csv", encoding="ISO-8859-1")
descriptions = pd.read_csv(f"{DATA_DIR}/product_descriptions.csv")
attributes = pd.read_csv(f"{DATA_DIR}/attributes.csv")
train = train.merge(descriptions, on="product_uid", how="left")

attr_lookup = (
    attributes.dropna(subset=["value"])
    .groupby("product_uid")["value"]
    .apply(lambda v: " ".join(v.astype(str)))
    .to_dict()
)

# ---------------------------------------------------------------------------
# Build extended feature matrix
# ---------------------------------------------------------------------------

corpus = pd.concat([train["product_title"], train["product_description"]]).astype(str)
tfidf = TfidfVectorizer(max_features=5000)
tfidf.fit(corpus)

X = np.array([
    build_feature_vector(row, attr_lookup, tfidf_vec=tfidf, stem=True)
    for _, row in train.iterrows()
])
y = train["relevance"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE
)

# ---------------------------------------------------------------------------
# Task 1 — Compare four regression models
# ---------------------------------------------------------------------------

MODELS = {
    "BaggingRF (baseline)": BaggingRegressor(
        estimator=RandomForestRegressor(n_estimators=15, max_depth=6, random_state=0, n_jobs=-1),
        n_estimators=40, max_samples=0.1, random_state=42, n_jobs=-1,
    ),
    "GradientBoosting": GradientBoostingRegressor(n_estimators=100, random_state=RANDOM_STATE),
    "Ridge": Ridge(),
    "KNeighbors": KNeighborsRegressor(n_neighbors=10, n_jobs=-1),
}

results = {}
for name, model in MODELS.items():
    t0 = time.time()
    model.fit(X_train, y_train)
    elapsed = time.time() - t0
    rmse = mean_squared_error(y_test, model.predict(X_test), squared=False)
    results[name] = {"RMSE": rmse, "Train time (s)": round(elapsed, 2)}
    print(f"{name:30s}  RMSE={rmse:.4f}  time={elapsed:.1f}s")

# ---------------------------------------------------------------------------
# Task 2 — Hyperparameter optimization on the best model
# ---------------------------------------------------------------------------

best_name = min(results, key=lambda k: results[k]["RMSE"])
print(f"\nBest model: {best_name} — tuning hyperparameters...")

# Example: tune GradientBoostingRegressor
param_dist = {
    "n_estimators": [50, 100, 200],
    "max_depth": [3, 4, 5, 6],
    "learning_rate": [0.05, 0.1, 0.2],
    "subsample": [0.7, 0.8, 1.0],
}

search = RandomizedSearchCV(
    GradientBoostingRegressor(random_state=RANDOM_STATE),
    param_distributions=param_dist,
    n_iter=20,
    cv=5,
    scoring="neg_root_mean_squared_error",
    random_state=RANDOM_STATE,
    n_jobs=-1,
    verbose=1,
)
search.fit(X_train, y_train)

best_rmse = mean_squared_error(y_test, search.best_estimator_.predict(X_test), squared=False)
print(f"Best params: {search.best_params_}")
print(f"Tuned RMSE:  {best_rmse:.4f}")

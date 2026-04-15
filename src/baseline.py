"""
Baseline replication — Week 9 & 10 tasks.

Replicates the method by Yao-Jen Chang (sklearn-random-forest).
Evaluates on an 80/20 split with RMSE. Also compares with/without stemming.
"""

import sys
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

sys.path.insert(0, ".")
from preprocessing import preprocess

DATA_DIR = "data"
RANDOM_STATE = 42  # fixed — keep the same split in every run

# ---------------------------------------------------------------------------
# Load & merge
# ---------------------------------------------------------------------------

train = pd.read_csv(f"{DATA_DIR}/train.csv", encoding="ISO-8859-1")
descriptions = pd.read_csv(f"{DATA_DIR}/product_descriptions.csv")
train = train.merge(descriptions, on="product_uid", how="left")

# ---------------------------------------------------------------------------
# Feature builder (Yao-Jen Chang style)
# ---------------------------------------------------------------------------

def common_words(a: str, b: str) -> int:
    return sum(1 for w in a.split() if w in b.split())


def build_baseline_features(df: pd.DataFrame, stem: bool = True) -> np.ndarray:
    df = df.copy()
    df["search_term"] = df["search_term"].map(
        lambda x: " ".join(preprocess(x, stem=stem))
    )
    df["product_title"] = df["product_title"].map(
        lambda x: " ".join(preprocess(x, stem=stem))
    )
    df["product_description"] = df["product_description"].map(
        lambda x: " ".join(preprocess(x, stem=stem))
    )
    return np.column_stack([
        df.apply(lambda r: common_words(r["search_term"], r["product_title"]), axis=1),
        df.apply(lambda r: common_words(r["search_term"], r["product_description"]), axis=1),
        df["search_term"].map(lambda x: len(x.split())),
    ])


# ---------------------------------------------------------------------------
# Train / test split
# ---------------------------------------------------------------------------

y = train["relevance"].values
X_stem = build_baseline_features(train, stem=True)
X_no_stem = build_baseline_features(train, stem=False)

X_tr, X_te, y_tr, y_te = train_test_split(X_stem, y, test_size=0.2, random_state=RANDOM_STATE)
X_tr_ns, X_te_ns, _, _ = train_test_split(X_no_stem, y, test_size=0.2, random_state=RANDOM_STATE)

# ---------------------------------------------------------------------------
# Model (Yao-Jen Chang baseline)
# ---------------------------------------------------------------------------

def make_model():
    rf = RandomForestRegressor(n_estimators=15, max_depth=6, random_state=0, n_jobs=-1)
    return BaggingRegressor(estimator=rf, n_estimators=40, max_samples=0.1,
                            random_state=42, n_jobs=-1)


model_stem = make_model()
model_stem.fit(X_tr, y_tr)
rmse_stem = mean_squared_error(y_te, model_stem.predict(X_te), squared=False)

model_no_stem = make_model()
model_no_stem.fit(X_tr_ns, y_tr)
rmse_no_stem = mean_squared_error(y_te_ns := model_no_stem.predict(X_te_ns),
                                  y_te, squared=False)

print(f"Baseline RMSE (with stemming): {rmse_stem:.4f}  (expected ~0.48)")
print(f"Baseline RMSE (no stemming):   {rmse_no_stem:.4f}")

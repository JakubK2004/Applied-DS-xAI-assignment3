"""
Feature weight inspection — Week 12 tasks.

Tasks:
1. Extract and rank feature importances from the best model
2. Plot top-N feature weights
3. Interpret the most important features
"""

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split

sys.path.insert(0, ".")
from features import FEATURE_NAMES_TFIDF

DATA_DIR = "../data"
RANDOM_STATE = 42

# ---------------------------------------------------------------------------
# Assumes X, y, and best_estimator are available from models.py
# Run models.py first, or re-build X/y here and fit a single model.
# ---------------------------------------------------------------------------

# Re-build features (import from models.py in practice)
from models import X_train, X_test, y_train, y_test, search  # noqa

best_model = search.best_estimator_

# ---------------------------------------------------------------------------
# Extract feature importances
# ---------------------------------------------------------------------------

importances = best_model.feature_importances_
feature_names = FEATURE_NAMES_TFIDF

# Sort descending
sorted_idx = np.argsort(importances)[::-1]
sorted_names = [feature_names[i] for i in sorted_idx]
sorted_values = importances[sorted_idx]

print("Feature importances (ranked):")
for name, val in zip(sorted_names, sorted_values):
    print(f"  {name:25s} {val:.4f}")

# ---------------------------------------------------------------------------
# Plot top-5 features
# ---------------------------------------------------------------------------

top_n = 5
plt.figure(figsize=(7, 4))
plt.barh(sorted_names[:top_n][::-1], sorted_values[:top_n][::-1])
plt.xlabel("Importance")
plt.title(f"Top-{top_n} feature importances")
plt.tight_layout()
plt.savefig("../results/figures/feature_importances.png")
plt.show()

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
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from features import FEATURE_NAMES_FULL

FIGURES_DIR = Path(__file__).parent.parent / "results" / "figures"

print("[1/4] Loading model and data from models.py...")
from models import X_train, X_test, y_train, y_test, search  # noqa
print("      Done.")

best_model = search.best_estimator_

# ---------------------------------------------------------------------------
# Extract feature importances
# ---------------------------------------------------------------------------

print("[2/4] Extracting feature importances...")
importances = best_model.feature_importances_
feature_names = FEATURE_NAMES_FULL

sorted_idx = np.argsort(importances)[::-1]
sorted_names = [feature_names[i] for i in sorted_idx]
sorted_values = importances[sorted_idx]

print("\nFeature importances (ranked):")
for name, val in zip(sorted_names, sorted_values):
    print(f"  {name:25s} {val:.4f}")

top_n = 5
plt.figure(figsize=(7, 4))
plt.barh(sorted_names[:top_n][::-1], sorted_values[:top_n][::-1])
plt.xlabel("Importance")
plt.title(f"Top-{top_n} feature importances")
plt.tight_layout()
plt.savefig(FIGURES_DIR / "feature_importances.png")
print(f"\n      Saved feature importances plot.")
plt.show()

# ---------------------------------------------------------------------------
# Feature correlation matrix
# ---------------------------------------------------------------------------

print("[3/4] Computing feature correlation matrix...")
df_features = pd.DataFrame(X_train, columns=FEATURE_NAMES_FULL)
corr = df_features.corr()

plt.figure(figsize=(10, 8))
plt.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
plt.colorbar()
plt.xticks(range(len(FEATURE_NAMES_FULL)), FEATURE_NAMES_FULL, rotation=90)
plt.yticks(range(len(FEATURE_NAMES_FULL)), FEATURE_NAMES_FULL)
plt.title("Feature correlation matrix")
plt.tight_layout()
plt.savefig(FIGURES_DIR / "feature_correlations.png")
print("      Saved correlation matrix plot.")
plt.show()

# ---------------------------------------------------------------------------
# Feature selection — retrain on top-N features only
# ---------------------------------------------------------------------------

print("[4/4] Retraining on selected features...")

# Adjust this to select more or fewer features
TOP_FEATURES = 8

selected_idx = sorted_idx[:TOP_FEATURES]
selected_names = [feature_names[i] for i in selected_idx]
print(f"      Selected features: {selected_names}")

X_train_sel = X_train[:, selected_idx]
X_test_sel  = X_test[:, selected_idx]

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import root_mean_squared_error

model_sel = GradientBoostingRegressor(**search.best_params_, random_state=42)
model_sel.fit(X_train_sel, y_train)

rmse_all = root_mean_squared_error(y_test, best_model.predict(X_test))
rmse_sel = root_mean_squared_error(y_test, model_sel.predict(X_test_sel))

print(f"\n  RMSE (all {len(feature_names)} features): {rmse_all:.4f}")
print(f"  RMSE (top {TOP_FEATURES} features):      {rmse_sel:.4f}")
print(f"  Difference:                  {rmse_sel - rmse_all:+.4f}")

print("\nDone.")

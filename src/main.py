"""
main.py — Home Depot Product Search Relevance
Applied Data Science and Explainable AI — Assignment 3

Run order:
  1. Data exploration
  2. Baseline replication + evaluation
  3. Model comparison + hyperparameter tuning
  4. Feature weight inspection

Usage:
  python main.py
"""

import sys
sys.path.insert(0, ".")

# ---------------------------------------------------------------------------
# Step 1 — Data exploration (Week 9)
# Loads train.csv, product_descriptions.csv, attributes.csv
# Answers Q1–Q6 and saves relevance distribution plot
# ---------------------------------------------------------------------------

# import data_exploration


# ---------------------------------------------------------------------------
# Step 2 — Baseline replication (Week 9 / 10)
# Replicates Yao-Jen Chang's random forest method
# Evaluates on 80/20 split with RMSE, with and without stemming
# ---------------------------------------------------------------------------

# import baseline


# ---------------------------------------------------------------------------
# Step 3 — Model comparison + hyperparameter tuning (Week 10 / 11)
# Builds extended feature matrix (overlap, TF-IDF, attributes, numbers/units)
# Compares 4 regression models on RMSE and training time
# Runs RandomizedSearchCV on the best model
# ---------------------------------------------------------------------------

# import models


# ---------------------------------------------------------------------------
# Step 4 — Feature weight inspection (Week 12)
# Extracts and ranks feature importances from the best tuned model
# Saves feature importance plot to results/figures/
# ---------------------------------------------------------------------------

# import feature_analysis

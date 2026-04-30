"""
Data exploration — Week 9 tasks.

Answers:
  Q1. Total number of product-query pairs in training data
  Q2. Number of unique products
  Q3. Two most occurring products
  Q4. Descriptive statistics for relevance values
  Q5. Distribution of relevance values (histogram)
  Q6. Top-5 most occurring brand names in attributes
"""

import matplotlib.pyplot as plt
from preprocessing import prepare_data_cached

train, test, attributes, attr_lookup = prepare_data_cached()

# ---------------------------------------------------------------------------
# Q1: Total product-query pairs
# ---------------------------------------------------------------------------

print("Q1 — Total product-query pairs:", len(train))

# ---------------------------------------------------------------------------
# Q2: Unique products
# ---------------------------------------------------------------------------

print("Q2 — Unique products:", train["product_uid"].nunique())

# ---------------------------------------------------------------------------
# Q3: Two most occurring products
# ---------------------------------------------------------------------------

print("\nQ3 — Two most occurring products:")
print(train["product_uid"].value_counts().head(2))

# ---------------------------------------------------------------------------
# Q4: Descriptive statistics for relevance
# ---------------------------------------------------------------------------

print("\nQ4 — Relevance statistics:")
print(train["relevance"].describe())

# ---------------------------------------------------------------------------
# Q5: Distribution of relevance values
# ---------------------------------------------------------------------------

plt.figure()
train["relevance"].hist(bins=20, edgecolor="black")
plt.title("Distribution of relevance scores")
plt.xlabel("Relevance")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("../results/figures/relevance_distribution.png")
# plt.show()

# ---------------------------------------------------------------------------
# Q6: Top-5 brand names in attributes
# ---------------------------------------------------------------------------

brands = attributes[attributes["name"] == "MFG Brand Name"]
print("\nQ6 — Top-5 brand names:")
print(brands["value"].value_counts().head(5))

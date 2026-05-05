# Data exploration: answers Q1–Q6 and saves the relevance distribution plot.

import matplotlib.pyplot as plt
from preprocessing import prepare_data_cached

train, test, attributes, attr_lookup = prepare_data_cached()

print("Q1 — Total product-query pairs:", len(train))
print("Q2 — Unique products:", train["product_uid"].nunique())

print("\nQ3 — Two most occurring products:")
print(train["product_uid"].value_counts().head(2))

print("\nQ4 — Relevance statistics:")
print(train["relevance"].describe())

# Q5: relevance distribution histogram
plt.figure()
train["relevance"].hist(bins=20, edgecolor="black")
plt.title("Distribution of relevance scores")
plt.xlabel("Relevance")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("../results/figures/relevance_distribution.png")
# plt.show()

brands = attributes[attributes["name"] == "MFG Brand Name"]
print("\nQ6 — Top-5 brand names:")
print(brands["value"].value_counts().head(5))

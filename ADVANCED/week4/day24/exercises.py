# Advanced Day 24 — Exercises

# Exercise 1 — Expand the lexicon
# Add 20 more positive and 20 more negative words to POS_WORDS/NEG_WORDS.
# Re-evaluate accuracy. By how much does it improve?
# TODO

# Exercise 2 — Bigram TF-IDF
# Modify the TF-IDF vectorizer to use (1,3) n-grams instead of (1,2).
# Compare accuracy. Does adding trigrams help or hurt?
# TODO

# Exercise 3 — Neutral detection
# Add a "NEUTRAL" class with 10 synthetic neutral reviews (factual statements,
# no opinion words). Extend the dataset to 3-class.
# Re-train TF-IDF+LR and report per-class precision/recall.
# TODO

# Exercise 4 — Confidence threshold
# For TF-IDF+LR, use predict_proba() to get confidence scores.
# Only classify reviews where confidence >= 0.7 as POSITIVE/NEGATIVE.
# Mark the rest as UNCERTAIN. What fraction is uncertain?
# TODO

# Exercise 5 — Domain adaptation
# Train on movie reviews (synthetic), test on product reviews (synthetic).
# This simulates domain shift. Compare accuracy vs training on product reviews.
# Print: cross-domain accuracy, in-domain accuracy, difference.
# TODO

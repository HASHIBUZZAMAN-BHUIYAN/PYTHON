# ~200 MB RAM, ~15s on CPU
"""
Day 23: Word Embeddings — Exercises
=====================================
Complete the 5 exercises below.
"""

import random
import numpy as np
# pip install gensim
from gensim.models import Word2Vec

random.seed(42)
np.random.seed(42)

# ─── Shared corpus for exercises ─────────────────────────────────────────────
RAW_SENTENCES = [
    "the king ruled the kingdom with wisdom",
    "the queen sat beside the king on the throne",
    "the prince will become king one day",
    "the princess and the prince live in the castle",
    "a man walked into the castle to meet the king",
    "a woman met the queen in the royal garden",
    "the boy played with the girl in the park",
    "the dog chased the cat through the forest",
    "the lion is the king of the jungle",
    "the tiger and the lion are powerful animals",
    "the cat sat on the mat and slept",
    "the dog ran fast across the green field",
    "the scientist studied the stars in the sky",
    "the engineer built a new computer program",
    "software and hardware work together in a computer",
    "the algorithm processes data very efficiently",
    "paris is the capital city of france",
    "london is a famous city in england",
    "italy and france are countries in europe",
    "rome was once ruled by a powerful emperor",
    "the doctor studied medicine and biology",
    "the teacher explained mathematics to students",
    "students learn science physics and chemistry",
    "the chef cooked delicious food in the kitchen",
    "apples and bananas are healthy fruits",
    "the ocean is deep and full of fish",
    "the mountain is tall and covered in snow",
    "birds fly high above the forest and ocean",
    "the sun rises in the east every morning",
    "stars and planets orbit in outer space",
] * 20  # repeat to give model enough data

CORPUS = [s.lower().split() for s in RAW_SENTENCES]

# ─────────────────────────────────────────────
# Exercise 1: Train and Inspect Word2Vec
# ─────────────────────────────────────────────
# TODO: Train a Word2Vec model (sg=1, vector_size=30, window=3,
#       min_count=2, epochs=30, seed=42) on CORPUS.
#
# Then:
#   a) Print the 5 most similar words to "king"
#   b) Print the 5 most similar words to "dog"
#   c) Print the cosine similarity between:
#      - "king" and "queen"
#      - "dog"  and "cat"
#      - "king" and "computer"
#   d) Print the size of the resulting vocabulary

# YOUR CODE HERE


# ─────────────────────────────────────────────
# Exercise 2: Analogy Solver
# ─────────────────────────────────────────────
# TODO: Using the trained Word2Vec model from Exercise 1,
# solve the following analogies using model.wv.most_similar(
#     positive=[...], negative=[...], topn=1):
#
#   a) king - man + woman = ?      (expected: queen)
#   b) prince - man + woman = ?    (expected: princess)
#   c) paris - france + italy = ?  (expected: rome)
#   d) dog - cat + lion = ?        (expected: tiger)
#
# For each analogy, print:
#   "{a} - {b} + {c} = {result}  (score: {score:.3f})"
# Handle KeyError gracefully if any word is not in vocabulary.

# YOUR CODE HERE


# ─────────────────────────────────────────────
# Exercise 3: Cosine Similarity from Scratch
# ─────────────────────────────────────────────
# TODO: Without using sklearn or gensim similarity methods,
# implement cosine_similarity(v1, v2) using only numpy.
# Formula: cos(v1,v2) = dot(v1,v2) / (||v1|| * ||v2||)
#
# Then compute and print similarities for these pairs:
#   ("king", "queen"), ("man", "woman"), ("dog", "computer"),
#   ("lion", "tiger"), ("ocean", "mountain")
#
# Compare your results to model.wv.similarity() output.
# They should match within floating-point precision.
# Note: use model.wv[word] to get the raw embedding vector.

# YOUR CODE HERE


# ─────────────────────────────────────────────
# Exercise 4: Document Embedding via Word Averaging
# ─────────────────────────────────────────────
# TODO: Represent each document as the average of its word vectors.
# This is called "average word embedding" or "doc embedding via mean pooling".
#
# Write a function `embed_document(model, text)` that:
#   1. Tokenizes text (lowercase split)
#   2. Looks up each token in model.wv (skip unknown words)
#   3. Returns the mean vector of all known word vectors
#   4. Returns a zero vector if no words are found
#
# Test on these 5 documents:
# docs = [
#     "the king and queen rule the kingdom",
#     "the prince and princess live in the castle",
#     "the dog and cat are friendly animals",
#     "the lion and tiger are wild animals",
#     "computers and software run algorithms",
# ]
# Print each doc's embedding (first 5 dims) and compute similarity
# between doc[0] and all others. Which is most similar to doc[0]?

# YOUR CODE HERE


# ─────────────────────────────────────────────
# Exercise 5: Odd-One-Out Detector
# ─────────────────────────────────────────────
# TODO: Use model.wv.doesnt_match(word_list) to find the odd word out
# from each group below. This method returns the word whose vector
# is furthest from the group centroid.
#
# groups = [
#     ["king", "queen", "prince", "dog"],
#     ["cat", "dog", "lion", "computer"],
#     ["paris", "london", "rome", "tiger"],
#     ["man", "woman", "boy", "ocean"],
# ]
#
# For each group, print:
#   Group: [...] → Odd one out: {word}
#
# Handle any words not in vocabulary by printing a warning and skipping.

# YOUR CODE HERE

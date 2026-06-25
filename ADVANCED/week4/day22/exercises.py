# ~150 MB RAM, ~10s on CPU
"""
Day 22: Text Preprocessing — Exercises
========================================
Complete the 5 exercises below.
Run this file to check your work (add print statements to verify output).
"""

import re
import math
from collections import Counter, defaultdict
import nltk
nltk.download('punkt',     quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet',   quiet=True)
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer

# ─────────────────────────────────────────────
# Exercise 1: Full Preprocessing Pipeline
# ─────────────────────────────────────────────
# TODO: Write a function `preprocess(text)` that:
#   1. Lowercases the text
#   2. Removes URLs (http/https) using regex
#   3. Removes all non-alphabetic characters (keep spaces)
#   4. Tokenizes with word_tokenize
#   5. Removes English stop words
#   6. Lemmatizes each token using WordNetLemmatizer (verb form, pos='v')
#   7. Returns a list of cleaned tokens
#
# Test it on these inputs and print the result:
#   text1 = "She was running quickly through the beautiful https://example.com forest!"
#   text2 = "The cats are eating fish while dogs are barking loudly"
#
# Expected output for text1 (approx): ['run', 'quickly', 'beautiful', 'forest']
# Expected output for text2 (approx): ['cat', 'eat', 'fish', 'dog', 'bark', 'loudly']

# YOUR CODE HERE


# ─────────────────────────────────────────────
# Exercise 2: Stemming vs Lemmatization Comparison
# ─────────────────────────────────────────────
# TODO: Given the word list below, print a comparison table with columns:
#   Word | Stem | Lemma(noun) | Lemma(verb)
# Use PorterStemmer for stemming and WordNetLemmatizer for lemmatization.
# Note: lemmatize(word, pos='n') for noun, lemmatize(word, pos='v') for verb.
#
# words = ["geese", "mice", "running", "better", "studies", "wolves", "went", "happily"]
#
# Print the table with aligned columns (use f-string formatting).

# YOUR CODE HERE


# ─────────────────────────────────────────────
# Exercise 3: Custom TF-IDF Vectorizer
# ─────────────────────────────────────────────
# TODO: Implement a class `SimpleTfidfVectorizer` with two methods:
#   - fit_transform(corpus): takes a list of strings, returns a 2D list (matrix)
#     where each row is a document and each column is a TF-IDF score.
#   - get_feature_names(): returns the sorted vocabulary list.
#
# TF formula:  tf(t, d) = count(t in d) / len(d)
# IDF formula: idf(t)   = log((N + 1) / (df(t) + 1)) + 1   (smoothed)
# TFIDF       = tf * idf
#
# Test with:
# docs = [
#     "python is great for data science",
#     "machine learning with python",
#     "data science and machine learning",
#     "python programming is fun",
# ]
# Print the top 3 words by TF-IDF score for each document.

# YOUR CODE HERE


# ─────────────────────────────────────────────
# Exercise 4: Bag-of-Words Cosine Similarity
# ─────────────────────────────────────────────
# TODO: Write a function `cosine_similarity(vec1, vec2)` that computes
# cosine similarity between two vectors (lists of numbers) from scratch.
# Formula: cos(A,B) = dot(A,B) / (||A|| * ||B||)
# Handle the zero-vector edge case (return 0.0).
#
# Then, using sklearn TfidfVectorizer, vectorize the 4 documents below and
# compute pairwise cosine similarity between all pairs. Print a similarity
# matrix with document labels.
#
# documents = [
#     "I love pizza and pasta",
#     "Pizza and pasta are my favorite foods",
#     "Machine learning is fascinating",
#     "Deep learning and neural networks are powerful",
# ]
# Expected: doc0 and doc1 should be most similar (~0.4-0.7),
#           doc2 and doc3 should be more similar to each other than to doc0/doc1.

# YOUR CODE HERE


# ─────────────────────────────────────────────
# Exercise 5: N-gram Language Model
# ─────────────────────────────────────────────
# TODO: Build a bigram frequency model from scratch.
#   1. Write a function `build_bigram_model(corpus)` where corpus is a list
#      of sentences. It should:
#      a. Tokenize each sentence (use .lower().split())
#      b. Add <START> and <END> tokens to each sentence
#      c. Count all bigrams (pairs of consecutive words)
#      d. Return a dict: {word: Counter({next_word: count, ...})}
#   2. Write `predict_next(model, word, top_n=3)` that returns the top_n
#      most likely next words given the current word.
#   3. Test on:
# corpus = [
#     "the cat sat on the mat",
#     "the cat ate the fish",
#     "the dog sat on the floor",
#     "the dog ate the bone",
#     "a cat and a dog are friends",
# ]
#   Print: predict_next(model, "the") → e.g., [('cat', 2), ('dog', 2), ('mat', 1)]
#   Print: predict_next(model, "cat") → top 3 next words after "cat"

# YOUR CODE HERE

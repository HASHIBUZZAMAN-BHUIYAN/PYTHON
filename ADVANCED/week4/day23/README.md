# Day 23: Word Embeddings

## Overview
Word embeddings represent words as dense numeric vectors where semantically similar words are close together in vector space. This is a massive improvement over Bag-of-Words which ignores meaning.

## Topics Covered
- Word2Vec intuition: CBOW vs Skip-gram architecture explained
- Training a tiny Word2Vec model with gensim on a synthetic corpus
- PCA visualization of word embeddings (2D scatter plot)
- Analogy arithmetic: king - man + woman ≈ queen

## Learning Objectives
By the end of this day you will be able to:
1. Explain the intuition behind CBOW and Skip-gram
2. Train a Word2Vec model with gensim on custom text
3. Find similar words and perform analogy reasoning
4. Visualize high-dimensional embeddings in 2D using PCA

## Files
| File | Purpose |
|------|---------|
| `lesson.py` | Word2Vec training, similarity, analogy, PCA visualization |
| `exercises.py` | 5 practice exercises (TODO stubs) |
| `solutions.py` | Full working solutions |
| `projects/project1_similar_words.py` | "Find similar words" tool |
| `projects/project2_analogy_solver.py` | Analogy solver (king-man+woman=queen) |
| `projects/project3_document_clustering.py` | Cluster documents via averaged word vectors |

## Setup
```
pip install gensim scikit-learn matplotlib
```

## Hardware Notes
- CPU-only, no GPU required
- Peak RAM: ~200 MB
- Estimated runtime: ~15 seconds (gensim trains fast on small corpus)

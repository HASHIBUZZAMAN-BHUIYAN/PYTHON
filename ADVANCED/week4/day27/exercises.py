# Advanced Day 27 — Exercises

# Exercise 1 — N-gram order comparison
# Build unigram, bigram, and trigram Markov models on the same corpus.
# Generate 20 words from each. Print all three and compare coherence.
# TODO

# Exercise 2 — Temperature sampling
# Add a "temperature" parameter to the bigram generator.
# At temperature=0.1 it picks the most common next word (greedy).
# At temperature=2.0 it flattens the distribution (more random).
# Use numpy to convert frequency counts to probabilities and apply temperature.
# TODO

# Exercise 3 — Top-N sentence extraction
# Modify extractive_summarize to also return the sentence scores as a dict.
# Visualize scores as a horizontal bar chart (sentence index vs score).
# TODO

# Exercise 4 — Sentence similarity summarization
# Implement a different summarization strategy: instead of scoring individual
# sentences by TF-IDF sum, score each sentence by its cosine similarity to
# the document-level centroid. Return top 3 sentences.
# TODO

# Exercise 5 — Mad-lib generator
# Build a template-based generator that fills in POS-tagged slots from a
# hand-crafted vocabulary: NOUN, VERB_PAST, ADJ, PLACE (20 words each).
# Generate 5 funny mad-lib sentences using the templates below.
# Templates:
#   "The {ADJ} {NOUN} {VERB_PAST} through {PLACE} and scared everyone."
#   "Yesterday, a {NOUN} {VERB_PAST} on a {ADJ} mountain in {PLACE}."
# TODO

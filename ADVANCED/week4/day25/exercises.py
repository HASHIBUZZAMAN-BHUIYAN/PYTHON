# Advanced Day 25 — Exercises

# Exercise 1 — Custom regex NER
# Add patterns for EMAIL and PHONE to ENTITY_PATTERNS.
# Test on: "Contact john.doe@company.com or call +1-555-867-5309 for info."
# TODO

# Exercise 2 — POS frequency chart
# Tag a 5-sentence paragraph with NLTK pos_tag.
# Count each simplified POS category (NOUN, VERB, ADJ, ADV, OTHER).
# Print a bar chart of the counts using matplotlib.
# TODO

# Exercise 3 — Entity co-occurrence
# Given a text with multiple sentences, build a co-occurrence matrix:
# For each pair of entity types, count how often they appear in the same sentence.
# Print the matrix as a table.
# TODO

# Exercise 4 — Proper noun extraction
# Use NLTK POS tags to extract all consecutive sequences of proper nouns (NNP, NNPS).
# These are likely to be multi-word named entities even without a NER model.
# Test on a 3-sentence tech news text.
# TODO

# Exercise 5 — Compare regex vs spaCy NER
# For the TEXT in lesson.py, compare what each approach finds.
# Print: entities only regex found, entities only spaCy found, entities both found.
# Fall back to regex-only if spaCy is unavailable.
# TODO

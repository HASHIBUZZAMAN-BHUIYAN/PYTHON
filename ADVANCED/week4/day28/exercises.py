# Advanced Day 28 — Exercises

# Exercise 1 — Expand the knowledge base
# Add 10 more Q&A pairs about any topic you like.
# Test the chatbot on 5 paraphrased versions of your new questions.
# What percentage does it answer correctly?
# TODO

# Exercise 2 — BM25 scoring
# Implement a simplified BM25 retrieval function (no library needed).
# BM25 formula: sum_t IDF(t) * (tf(t,d)*(k+1)) / (tf(t,d) + k*(1-b+b*dl/avgdl))
# Use k=1.5, b=0.75. Compare BM25 top-1 accuracy vs TF-IDF cosine on 5 queries.
# TODO

# Exercise 3 — Multi-turn memory
# Add a session variable that stores the last 3 (query, answer) pairs.
# Include the last answer in the context when vectorizing the current query.
# Show how this helps with follow-up questions like "explain that further".
# TODO

# Exercise 4 — Confidence calibration
# Run 20 queries (10 in-KB, 10 out-of-domain).
# Plot a histogram of similarity scores for both groups.
# Find the threshold that maximizes (true positive - false positive) rate.
# TODO

# Exercise 5 — Response templating
# Modify the chatbot to insert the query topic into response templates.
# Example: "Great question about {topic}! {answer}"
# Extract the topic by finding the longest noun phrase in the query (regex).
# TODO

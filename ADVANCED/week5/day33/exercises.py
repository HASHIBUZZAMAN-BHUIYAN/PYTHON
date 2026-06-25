# Advanced Day 33 — Exercises

# Exercise 1 — Temperature scaling
# Add a "temperature" parameter to scaled_dot_product:
#   scores = Q @ K.T / (sqrt(d_k) * temperature)
# At temperature=0.1, attention becomes spiky (one token dominates).
# At temperature=5.0, attention becomes uniform.
# Visualize both for a 5-token sequence.
# TODO

# Exercise 2 — Cross-attention
# Implement cross-attention: encoder output is K,V; decoder query is Q.
# Encoder: 4 tokens of dim 8.  Decoder: 3 tokens of dim 8.
# Show output shape and attention weights shape.
# TODO

# Exercise 3 — Implement MultiHeadAttention from scratch
# Without using nn.MultiheadAttention, implement it:
#   - Project Q, K, V to h heads using h linear layers each
#   - Compute attention per head
#   - Concatenate and project back
# Test with h=2, d_model=8, seq=4.
# TODO

# Exercise 4 — Attention on embeddings
# Create a simple integer → embedding lookup (10 tokens, dim 16).
# Run self-attention on a 5-token input sequence.
# Print which tokens attend most to each other (highest weight pairs).
# TODO

# Exercise 5 — Relative attention weights
# After computing attention weights, normalize each column by its entropy
# H = -sum(w * log(w + 1e-9)) to measure how "focused" each query is.
# Print entropy per query position. Low entropy = focused attention.
# TODO

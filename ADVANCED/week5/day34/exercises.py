# Advanced Day 34 — Exercises

# Exercise 1 — Learned vs sinusoidal PE
# Add a "learned" positional encoding: nn.Embedding(max_len, d_model).
# Train the TinyTransformer with learned PE vs sinusoidal PE.
# Compare final accuracy. Which converges faster?
# TODO

# Exercise 2 — Ablation: no residuals
# Remove residual connections from TransformerBlock (remove the "+x" terms).
# Train on the same task. Compare accuracy and loss curves.
# Explain what happens to gradient flow.
# TODO

# Exercise 3 — Ablation: no LayerNorm
# Remove LayerNorm from TransformerBlock.
# Train and compare. Does training become unstable?
# TODO

# Exercise 4 — Depth vs width
# Fix total params ~2000. Compare:
#   - Wide:   1 block, d_model=32, n_heads=4, ff=64
#   - Deep:   4 blocks, d_model=8,  n_heads=2, ff=16
# Which reaches higher accuracy on the 3-class token classification task?
# TODO

# Exercise 5 — Causal transformer (language model style)
# Add a causal mask to each TransformerBlock so token i only attends to 1..i.
# Generate next-token predictions: for an 8-token sequence, predict token 9.
# Use argmax over vocabulary as the next token. Print 5 predicted sequences.
# TODO

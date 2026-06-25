# Day 29: RNN Fundamentals

## Learning Objectives
- Understand sequence data and why order matters
- Build a vanilla RNN from scratch using NumPy
- Implement the same RNN using PyTorch nn.RNN
- Observe and understand the vanishing gradient problem

## Topics Covered
1. **Sequence Data Intuition** — time series, text, audio — data where position matters
2. **Vanilla RNN from Scratch** — forward pass and BPTT with NumPy
3. **PyTorch nn.RNN** — clean, production-style implementation
4. **Vanishing Gradient Demo** — visualize gradient magnitudes across timesteps

## Files
- `lesson.py` — main teaching file with all concepts
- `exercises.py` — 5 practice exercises (TODOs only)
- `solutions.py` — full working solutions
- `projects/project1_char_predictor.py` — character-level next-char predictor
- `projects/project2_timeseries_rnn.py` — sine-wave time series prediction
- `projects/project3_seq_classifier.py` — classify sequences by pattern

## Hardware
- CPU-only, ~150-250 MB RAM
- Runtime: ~30-60 seconds per project on Ryzen 7

## Key Concepts
```
h_t = tanh(W_hh * h_{t-1} + W_xh * x_t + b)
y_t = W_hy * h_t + b_y
```

The hidden state `h_t` carries memory from previous timesteps.
Gradients must flow back through every timestep — this causes vanishing/exploding gradients.

## Run Order
```
python lesson.py
python exercises.py   # attempt first
python solutions.py   # check answers
python projects/project1_char_predictor.py
python projects/project2_timeseries_rnn.py
python projects/project3_seq_classifier.py
```

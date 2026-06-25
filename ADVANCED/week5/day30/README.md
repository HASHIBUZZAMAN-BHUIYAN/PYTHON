# Day 30: LSTM & GRU

## Learning Objectives
- Understand why vanilla RNNs fail on long sequences
- Learn LSTM cell gates (forget / input / output) intuitively
- Understand GRU as a simplified alternative to LSTM
- Use PyTorch nn.LSTM and nn.GRU
- Compare RNN vs LSTM performance on long-dependency tasks

## Topics Covered
1. **Vanishing Gradient Recap** — why RNN memory fades
2. **LSTM Gates** — cell state as the "memory highway"
3. **GRU** — fewer parameters, similar performance
4. **PyTorch API** — nn.LSTM, nn.GRU usage patterns
5. **Comparison** — RNN vs LSTM on a held-out long-sequence task

## Files
- `lesson.py` — main teaching file
- `exercises.py` — 5 practice exercises (TODOs)
- `solutions.py` — full working solutions
- `projects/project1_char_lstm.py` — character-level text generator with LSTM
- `projects/project2_timeseries_lstm.py` — stock-like time series forecasting
- `projects/project3_sentiment_lstm.py` — sentiment classification + baseline comparison

## Hardware
- CPU-only, ~150-300 MB RAM
- Runtime: ~30-90 seconds per project on Ryzen 7

## Key Equations
```
LSTM:
  f_t = σ(W_f · [h_{t-1}, x_t] + b_f)      ← forget gate
  i_t = σ(W_i · [h_{t-1}, x_t] + b_i)      ← input gate
  g_t = tanh(W_g · [h_{t-1}, x_t] + b_g)   ← candidate cell
  o_t = σ(W_o · [h_{t-1}, x_t] + b_o)      ← output gate
  c_t = f_t ⊙ c_{t-1} + i_t ⊙ g_t          ← cell state update
  h_t = o_t ⊙ tanh(c_t)                    ← hidden state

GRU (simpler):
  r_t = σ(W_r · [h_{t-1}, x_t])            ← reset gate
  z_t = σ(W_z · [h_{t-1}, x_t])            ← update gate
  h̃_t = tanh(W · [r_t ⊙ h_{t-1}, x_t])   ← candidate
  h_t = (1 - z_t) ⊙ h_{t-1} + z_t ⊙ h̃_t ← new hidden
```

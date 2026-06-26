# DL Reference Folder

Drop-in training loop templates for deep learning.

| File | Contents |
|------|----------|
| `pytorch_training_loop_template.py` | Full PyTorch loop with callbacks, scheduler, early-stop |
| `keras_training_loop_template.py`   | Full Keras/TF loop with callbacks, plot history |
| `lstm_template.py`                  | LSTM/GRU for time-series and text sequence tasks |
| `autoencoder_template.py`           | Autoencoder for compression, denoising, anomaly detection |
| `tiny_transformer_template.py`      | Minimal transformer block from scratch (d_model=32) |

## Related lessons
- ADVANCED/week2/day08 — NN from scratch
- ADVANCED/week2/day09 — PyTorch intro
- ADVANCED/week2/day10 — TensorFlow/Keras
- ADVANCED/week5/day29 — RNN fundamentals
- ADVANCED/week5/day30 — LSTM & GRU
- ADVANCED/week5/day31 — Autoencoders
- ADVANCED/week5/day33 — Attention mechanism
- ADVANCED/week5/day34 — Transformer basics

---

## Runnable Projects

All commands are run from the `PYTHON/` folder using the venv Python.
Outputs (PNG plots) are saved to `DL/outputs/`.

| File | Description | Run command |
|------|-------------|-------------|
| `road_traffic_time_series.py` | LSTM forecasting on synthetic hourly traffic data; MAE/RMSE vs actual | `python DL\road_traffic_time_series.py` |
| `human_activity_recognition.py` | 1D-CNN classifies walking/running/sitting from synthetic accelerometer windows | `python DL\human_activity_recognition.py` |
| `human_text_understanding.py` | Trainable embedding + dense NN classifies urgent vs non-urgent support messages | `python DL\human_text_understanding.py` |
| `road_sign_recognition.py` | CNN classifies 4 synthetic traffic sign shapes (stop/yield/warning/speed) | `python DL\road_sign_recognition.py` |
| `anomaly_detection_realworld.py` | 1D-conv autoencoder detects anomaly spikes in synthetic industrial sensor data | `python DL\anomaly_detection_realworld.py` |
| `model_comparison_realworld.py` | Compares naive/moving-avg/linear-reg/LSTM on traffic data; explains when DL helps | `python DL\model_comparison_realworld.py` |

> **Hardware note:** All scripts run CPU-only (no GPU). Tested on Ryzen 7, 8 GB RAM.
> Peak RAM ~300 MB. Slowest script (traffic LSTM) takes ~10s. All data is synthetic.

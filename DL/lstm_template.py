# DL Reference — LSTM/GRU Template
# For time-series forecasting and sequence classification.
# ~50 MB RAM, <5s on CPU
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# ─── 1. LSTM MODEL ────────────────────────────────────────────────────────────
class LSTMModel(nn.Module):
    """Stacked LSTM for sequence tasks."""
    def __init__(self, input_size, hidden_size, num_layers, output_size,
                 dropout=0.2, bidirectional=False):
        super().__init__()
        self.hidden_size   = hidden_size
        self.num_layers    = num_layers
        self.bidirectional = bidirectional
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers,
            batch_first=True, dropout=dropout if num_layers>1 else 0.,
            bidirectional=bidirectional,
        )
        d = 2 if bidirectional else 1
        self.fc = nn.Linear(hidden_size * d, output_size)

    def forward(self, x, h=None):
        out, (hn, cn) = self.lstm(x, h)
        return self.fc(out)   # (batch, seq, output_size)

class GRUModel(nn.Module):
    """GRU — simpler alternative to LSTM."""
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.2):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers,
                          batch_first=True, dropout=dropout if num_layers>1 else 0.)
        self.fc  = nn.Linear(hidden_size, output_size)

    def forward(self, x, h=None):
        out, _ = self.gru(x, h)
        return self.fc(out)

# ─── 2. SEQUENCE DATASET BUILDER ─────────────────────────────────────────────
def make_seq_dataset(series: np.ndarray, seq_len: int, forecast_steps=1):
    """
    Sliding window dataset from a 1D time series.
    Returns: X (N, seq_len, 1), y (N, forecast_steps)
    """
    X, y = [], []
    for i in range(len(series) - seq_len - forecast_steps + 1):
        X.append(series[i : i+seq_len])
        y.append(series[i+seq_len : i+seq_len+forecast_steps])
    X = np.array(X, dtype=np.float32).reshape(-1, seq_len, 1)
    y = np.array(y, dtype=np.float32)
    return torch.from_numpy(X), torch.from_numpy(y)

# ─── 3. TRAINING LOOP ─────────────────────────────────────────────────────────
def train_lstm(model, X_train, y_train, X_val, y_val,
               n_epochs=30, batch_size=32, lr=1e-3, patience=5):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    train_ds  = torch.utils.data.TensorDataset(X_train, y_train)
    train_dl  = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    best_loss = float("inf"); wait = 0
    history   = {"train": [], "val": []}

    for epoch in range(1, n_epochs+1):
        model.train(); total = 0.
        for xb, yb in train_dl:
            pred = model(xb)[:, -1, :]   # last time-step output
            loss = criterion(pred, yb)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            total += loss.item()*len(xb)
        train_loss = total / len(train_dl.dataset)

        model.eval()
        with torch.no_grad():
            val_pred = model(X_val)[:, -1, :]
            val_loss = criterion(val_pred, y_val).item()
        history["train"].append(train_loss); history["val"].append(val_loss)

        if epoch % 5 == 0:
            print(f"  Epoch {epoch:>3}: train={train_loss:.4f}  val={val_loss:.4f}")
        if val_loss < best_loss - 1e-5:
            best_loss=val_loss; wait=0
            torch.save(model.state_dict(), "best_lstm.pt")
        else:
            wait += 1
            if wait >= patience:
                print(f"  Early stop at epoch {epoch}"); break
    return history

# ─── DEMO ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    torch.manual_seed(42); np.random.seed(42)
    t = np.linspace(0, 10*np.pi, 600)
    series = (np.sin(t) + 0.5*np.sin(2*t) + np.random.normal(0,0.05,len(t))).astype(np.float32)
    # Normalize
    series = (series - series.mean()) / series.std()
    SEQ=20; split=480
    X, y = make_seq_dataset(series, SEQ, forecast_steps=1)
    X_tr, y_tr = X[:split], y[:split]
    X_vl, y_vl = X[split:], y[split:]

    model = LSTMModel(input_size=1, hidden_size=32, num_layers=1, output_size=1)
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")
    hist  = train_lstm(model, X_tr, y_tr, X_vl, y_vl, n_epochs=30, batch_size=32)

    model.load_state_dict(torch.load("best_lstm.pt", map_location="cpu"))
    model.eval()
    with torch.no_grad():
        pred = model(X_vl)[:,-1,0].numpy()
    actual = y_vl[:,0].numpy()
    rmse   = np.sqrt(np.mean((pred-actual)**2))
    print(f"Val RMSE: {rmse:.4f}")

    plt.figure(figsize=(10,3))
    plt.plot(actual, label="Actual"); plt.plot(pred, label="Predicted", alpha=0.8)
    plt.title("LSTM Time Series Forecast"); plt.legend(); plt.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig("lstm_forecast.png",dpi=80); plt.close()
    print("Saved lstm_forecast.png")

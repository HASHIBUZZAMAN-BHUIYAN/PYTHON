"""
Project: Sequence Forecasting with LSTM vs Transformer
Teaches: time series prediction, rolling window approach,
         comparing LSTM and Transformer on same forecasting task.
~80 MB RAM, ~10s on CPU
"""
import numpy as np
import torch, torch.nn as nn, torch.optim as optim
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

torch.manual_seed(42); np.random.seed(42)

# ─── Generate time series: 3 superimposed sine waves ──────────────────────────
T = 2000
t = np.linspace(0, 40*np.pi, T)
signal = (np.sin(t) + 0.5*np.sin(3*t) + 0.25*np.sin(7*t) + 0.1*np.random.randn(T)).astype(np.float32)
signal = (signal - signal.mean()) / (signal.std() + 1e-8)

# ─── Rolling window dataset ───────────────────────────────────────────────────
SEQ_LEN=32; HORIZON=1
def make_seq_data(sig,seq_len,horizon):
    X,y=[],[]
    for i in range(len(sig)-seq_len-horizon+1):
        X.append(sig[i:i+seq_len]); y.append(sig[i+seq_len:i+seq_len+horizon])
    return (torch.tensor(np.array(X)).unsqueeze(-1),  # [N,seq,1]
            torch.tensor(np.array(y)))                 # [N,horizon]

X_all, y_all = make_seq_data(signal, SEQ_LEN, HORIZON)
split = int(len(X_all)*0.8)
X_tr,y_tr=X_all[:split],y_all[:split]; X_te,y_te=X_all[split:],y_all[split:]

# ─── Models ──────────────────────────────────────────────────────────────────
class LSTMForecast(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm=nn.LSTM(1,32,num_layers=2,batch_first=True,dropout=0.1)
        self.fc  =nn.Linear(32,HORIZON)
    def forward(self,x): _,(h,_)=self.lstm(x); return self.fc(h[-1])

class TransformerForecast(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_proj=nn.Linear(1,16)
        self.mha=nn.MultiheadAttention(16,4,batch_first=True)
        self.ff =nn.Sequential(nn.Linear(16,32),nn.GELU(),nn.Linear(32,16))
        self.n1=nn.LayerNorm(16); self.n2=nn.LayerNorm(16)
        self.fc=nn.Linear(16,HORIZON)
    def forward(self,x):
        e=self.input_proj(x)
        a,_=self.mha(e,e,e); e=self.n1(e+a); e=self.n2(e+self.ff(e))
        return self.fc(e[:,-1])  # last position

import time
results = {}
for name, model in [("LSTM",LSTMForecast()), ("Transformer",TransformerForecast())]:
    opt=optim.Adam(model.parameters(),lr=1e-3); crit=nn.MSELoss()
    t0=time.time(); losses=[]
    for epoch in range(200):
        idx=torch.randperm(len(X_tr))[:64]
        loss=crit(model(X_tr[idx]),y_tr[idx])
        opt.zero_grad(); loss.backward(); opt.step(); losses.append(loss.item())
    elapsed=time.time()-t0
    model.eval()
    with torch.no_grad():
        preds=model(X_te).numpy().flatten()
        test_mse=((preds-y_te.numpy().flatten())**2).mean()
    results[name]={"losses":losses,"preds":preds,"mse":test_mse,"time":elapsed}
    print(f"  {name}: test_mse={test_mse:.5f}  time={elapsed:.1f}s")

# ─── Visualize ────────────────────────────────────────────────────────────────
true_vals=y_te.numpy().flatten()
fig,axes=plt.subplots(3,1,figsize=(12,9))
axes[0].plot(results["LSTM"]["losses"],label="LSTM",color="steelblue",alpha=0.8)
axes[0].plot(results["Transformer"]["losses"],label="Transformer",color="tomato",alpha=0.8)
axes[0].set_title("Training Loss"); axes[0].legend(); axes[0].set_xlabel("Step")

N_SHOW=200
axes[1].plot(true_vals[:N_SHOW],label="True",color="k",linewidth=1.5)
axes[1].plot(results["LSTM"]["preds"][:N_SHOW],label=f"LSTM (MSE={results['LSTM']['mse']:.5f})",color="steelblue",alpha=0.8)
axes[1].set_title("LSTM Forecast vs True"); axes[1].legend(); axes[1].grid(alpha=0.3)

axes[2].plot(true_vals[:N_SHOW],label="True",color="k",linewidth=1.5)
axes[2].plot(results["Transformer"]["preds"][:N_SHOW],label=f"Transformer (MSE={results['Transformer']['mse']:.5f})",color="tomato",alpha=0.8)
axes[2].set_title("Transformer Forecast vs True"); axes[2].legend(); axes[2].grid(alpha=0.3)

plt.suptitle("Sequence Forecasting: LSTM vs Transformer",fontsize=11)
plt.tight_layout(); plt.savefig("seq_forecast.png",dpi=85); plt.close(); print("Saved seq_forecast.png")

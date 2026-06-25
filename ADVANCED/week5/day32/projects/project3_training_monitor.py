"""
Project: GAN Training Monitor — Detecting Collapse and Instability
Teaches: tracking D(real), D(fake), gradient norms, and mode coverage
         to diagnose common GAN training problems.
~50 MB RAM, ~10s on CPU
"""
import numpy as np
import torch, torch.nn as nn, torch.optim as optim
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

torch.manual_seed(42); np.random.seed(42)

MEANS=[(2,2),(-2,2),(2,-2),(-2,-2)]
def sample_real(n=128):
    data=[]
    for _ in range(n):
        mu=MEANS[np.random.randint(4)]; data.append(np.array(mu)+0.4*np.random.randn(2))
    return torch.tensor(np.array(data,dtype=np.float32))

Z=8
class G(nn.Module):
    def __init__(self): super().__init__(); self.net=nn.Sequential(nn.Linear(Z,32),nn.ReLU(),nn.Linear(32,32),nn.ReLU(),nn.Linear(32,2))
    def forward(self,z): return self.net(z)
class D(nn.Module):
    def __init__(self): super().__init__(); self.net=nn.Sequential(nn.Linear(2,32),nn.LeakyReLU(0.2),nn.Linear(32,32),nn.LeakyReLU(0.2),nn.Linear(32,1),nn.Sigmoid())
    def forward(self,x): return self.net(x)

def count_modes(G_net, threshold=0.8):
    with torch.no_grad():
        pts=G_net(torch.randn(512,Z)).numpy()
    modes_covered=0
    for mu in MEANS:
        dists=np.sqrt(((pts-np.array(mu))**2).sum(-1))
        if (dists<threshold).mean()>0.05: modes_covered+=1
    return modes_covered

def get_grad_norm(model):
    total=0.0
    for p in model.parameters():
        if p.grad is not None: total+=p.grad.data.norm(2).item()**2
    return total**0.5

metrics={"g_loss":[],"d_loss":[],"d_real":[],"d_fake":[],"g_grad":[],"d_grad":[],"modes":[]}
gen=G(); dis=D()
opt_G=optim.Adam(gen.parameters(),lr=2e-4,betas=(0.5,0.999))
opt_D=optim.Adam(dis.parameters(),lr=2e-4,betas=(0.5,0.999))
bce=nn.BCELoss(); EPOCHS=500; BATCH=64

print("Training GAN with full monitoring ...")
for epoch in range(EPOCHS):
    xr=sample_real(BATCH); z=torch.randn(BATCH,Z); xf=gen(z).detach()
    d_r=dis(xr).mean().item(); d_f=dis(xf).mean().item()
    ld=bce(dis(xr),torch.ones(BATCH,1))+bce(dis(xf),torch.zeros(BATCH,1))
    opt_D.zero_grad(); ld.backward(); dg=get_grad_norm(dis); opt_D.step()

    z=torch.randn(BATCH,Z); xf=gen(z)
    lg=bce(dis(xf),torch.ones(BATCH,1))
    opt_G.zero_grad(); lg.backward(); gg=get_grad_norm(gen); opt_G.step()

    metrics["g_loss"].append(lg.item()); metrics["d_loss"].append(ld.item())
    metrics["d_real"].append(d_r);       metrics["d_fake"].append(d_f)
    metrics["g_grad"].append(gg);        metrics["d_grad"].append(dg)
    if epoch%20==0: metrics["modes"].append(count_modes(gen))
    if (epoch+1)%100==0:
        m=metrics["modes"][-1] if metrics["modes"] else "?"
        print(f"  Epoch {epoch+1}  G={lg.item():.3f}  D={ld.item():.3f}  D(real)={d_r:.2f}  D(fake)={d_f:.2f}  modes={m}/4")

# ─── Diagnosis ────────────────────────────────────────────────────────────────
final_modes=count_modes(gen)
d_real_avg =np.mean(metrics["d_real"][-50:])
d_fake_avg =np.mean(metrics["d_fake"][-50:])
print(f"\n=== Training Diagnosis ===")
print(f"  Final modes covered    : {final_modes}/4")
print(f"  D(real) last 50 steps  : {d_real_avg:.3f}  (healthy ≈ 0.5-0.7)")
print(f"  D(fake) last 50 steps  : {d_fake_avg:.3f}  (healthy ≈ 0.3-0.5)")
if d_real_avg>0.95: print("  WARNING: D_real>0.95 — discriminator may be too strong")
if d_fake_avg>0.7:  print("  WARNING: D_fake>0.7 — generator may have collapsed")
if final_modes<3:   print("  WARNING: Mode collapse — only {final_modes}/4 modes covered")
else:               print("  OK: Generator covers all 4 modes")

# ─── Visualize ────────────────────────────────────────────────────────────────
fig,axes=plt.subplots(2,3,figsize=(13,7))
x_ax=range(EPOCHS)
axes[0,0].plot(x_ax,metrics["g_loss"],alpha=0.8,color="tomato"); axes[0,0].set_title("G Loss")
axes[0,1].plot(x_ax,metrics["d_loss"],alpha=0.8,color="steelblue"); axes[0,1].set_title("D Loss")
axes[0,2].plot(x_ax,metrics["d_real"],label="D(real)",color="green")
axes[0,2].plot(x_ax,metrics["d_fake"],label="D(fake)",color="orange")
axes[0,2].axhline(0.5,color="k",linestyle="--",linewidth=0.8); axes[0,2].set_title("D Outputs"); axes[0,2].legend()
axes[1,0].plot(x_ax,metrics["g_grad"],alpha=0.8,color="tomato"); axes[1,0].set_title("G Gradient Norm")
axes[1,1].plot(x_ax,metrics["d_grad"],alpha=0.8,color="steelblue"); axes[1,1].set_title("D Gradient Norm")
axes[1,2].plot(range(0,EPOCHS,20),metrics["modes"],"o-",color="purple"); axes[1,2].set_ylim(0,5)
axes[1,2].set_title("Modes Covered"); axes[1,2].axhline(4,color="k",linestyle="--")
plt.suptitle("GAN Training Monitor",fontsize=11); plt.tight_layout()
plt.savefig("gan_monitor.png",dpi=85); plt.close(); print("Saved gan_monitor.png")

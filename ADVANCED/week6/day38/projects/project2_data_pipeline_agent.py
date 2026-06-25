"""
Project: Data Analysis Pipeline Agent
Teaches: chaining data tools (load, clean, transform, analyze, visualize)
         with error recovery and intermediate result passing.
~30 MB RAM, ~1s on CPU
"""
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

class ToolError(Exception): pass

# ─── Data tools ───────────────────────────────────────────────────────────────
def tool_load_data(source):
    """Simulate loading a dataset."""
    np.random.seed(42)
    if source == "sales":
        return {"months": list(range(1,13)),
                "revenue": (100 + 20*np.sin(np.arange(12)*0.5) + 5*np.random.randn(12)).tolist(),
                "units":   (50 + 10*np.random.randn(12)).tolist()}
    if source == "sensors":
        t = np.linspace(0,10,200)
        return {"time": t.tolist(),
                "temp": (20+5*np.sin(t)+np.random.randn(200)).tolist(),
                "pressure": (1013+2*np.cos(t)+0.5*np.random.randn(200)).tolist()}
    raise ToolError(f"Unknown data source: {source}")

def tool_clean_data(data, drop_outliers=True):
    """Remove outliers beyond 3 standard deviations."""
    cleaned = {}
    for key, vals in data.items():
        if not isinstance(vals[0], (int,float)): cleaned[key]=vals; continue
        arr = np.array(vals); mu=arr.mean(); std=arr.std()
        if drop_outliers:
            mask = np.abs(arr-mu) < 3*std
            cleaned[key] = arr[mask].tolist()
        else:
            cleaned[key] = vals
    return cleaned

def tool_normalize(data, method="minmax"):
    result = {}
    for key, vals in data.items():
        if not isinstance(vals[0], (int,float)): result[key]=vals; continue
        arr = np.array(vals, dtype=float)
        if method == "minmax":
            mn,mx = arr.min(), arr.max()
            result[key] = ((arr-mn)/(mx-mn+1e-8)).tolist()
        elif method == "zscore":
            result[key] = ((arr-arr.mean())/(arr.std()+1e-8)).tolist()
        else: raise ToolError(f"Unknown normalization: {method}")
    return result

def tool_stats(data):
    stats = {}
    for key, vals in data.items():
        if not isinstance(vals[0], (int,float)): continue
        arr = np.array(vals)
        stats[key] = {"mean":float(arr.mean()), "std":float(arr.std()),
                      "min":float(arr.min()), "max":float(arr.max()),
                      "n":len(arr)}
    return stats

def tool_plot(data, title="Data Plot", save_path="pipeline_plot.png"):
    fig, ax = plt.subplots(figsize=(8,4))
    for key, vals in data.items():
        if isinstance(vals[0], (int,float)): ax.plot(vals, label=key, alpha=0.8)
    ax.set_title(title); ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(save_path, dpi=80); plt.close()
    return save_path

# ─── Pipeline agent ───────────────────────────────────────────────────────────
class PipelineAgent:
    TOOLS = {"load":tool_load_data,"clean":tool_clean_data,
             "normalize":tool_normalize,"stats":tool_stats,"plot":tool_plot}

    def __init__(self):
        self.data    = {}
        self.log     = []
        self.n_errors= 0

    def run_step(self, step_name, tool_name, **kwargs):
        print(f"\n  [{step_name}] → {tool_name}({list(kwargs.keys())})")
        try:
            fn = self.TOOLS[tool_name]
            result = fn(**kwargs)
            self.log.append({"step":step_name,"tool":tool_name,"ok":True})
            return result
        except ToolError as e:
            self.n_errors += 1
            self.log.append({"step":step_name,"tool":tool_name,"ok":False,"error":str(e)})
            print(f"  ERROR: {e}")
            return None

    def analyze(self, source):
        print(f"=== Data Pipeline: {source} ===")
        raw  = self.run_step("load",    "load",      source=source)
        if raw is None: return
        cln  = self.run_step("clean",   "clean",     data=raw)
        norm = self.run_step("normalize","normalize", data=cln, method="minmax")
        st   = self.run_step("stats",   "stats",     data=norm)
        path = self.run_step("plot",    "plot",       data=norm, title=f"{source} (normalized)", save_path=f"{source}_pipeline.png")

        print(f"\n  Stats:")
        for col, s in (st or {}).items():
            print(f"    {col}: mean={s['mean']:.3f}  std={s['std']:.3f}  n={s['n']}")
        print(f"  Plot saved: {path}")

agent = PipelineAgent()
agent.analyze("sales")
agent.analyze("sensors")

print(f"\n=== Pipeline Summary ===")
print(f"  Steps executed: {len(agent.log)}")
print(f"  Errors: {agent.n_errors}")
for entry in agent.log:
    status = "✓" if entry["ok"] else "✗"
    print(f"  {status} {entry['step']}: {entry['tool']}")

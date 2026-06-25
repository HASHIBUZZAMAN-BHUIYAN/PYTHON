"""
Project: Full Curriculum Integration Demo
Teaches: end-to-end showcase integrating all major curriculum modules:
  ML (sklearn) + NLP (TF-IDF) + DL (PyTorch) + Agent (ToolRegistry)
  + Evaluation + Guardrails in a single runnable demo.
~60 MB RAM, ~5s on CPU
"""
import re, time, random
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import Counter

print("="*65)
print("  FULL CURRICULUM INTEGRATION DEMO — 42-Day Python AI Course")
print("="*65)

# ─── MODULE 1: ML (Week 1-3) ──────────────────────────────────────────────────
print("\n[MODULE 1] Machine Learning — Sklearn Pipeline")
from sklearn.datasets import make_classification
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

X, y = make_classification(n_samples=500, n_features=10, n_informative=5, random_state=42)
pipe  = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=200))])
cv    = cross_val_score(pipe, X, y, cv=5, scoring="accuracy")
pipe.fit(X, y)
print(f"  Logistic Regression CV: {cv.mean():.3f} ± {cv.std():.3f}")
sample = X[:3]
preds  = pipe.predict(sample)
print(f"  Predictions on 3 samples: {preds.tolist()}")

# ─── MODULE 2: PyTorch MLP (Week 3) ──────────────────────────────────────────
print("\n[MODULE 2] Deep Learning — PyTorch MLP")
try:
    import torch, torch.nn as nn

    class MLP(nn.Module):
        def __init__(self): super().__init__(); self.net=nn.Sequential(nn.Linear(10,32),nn.ReLU(),nn.Linear(32,1),nn.Sigmoid())
        def forward(self,x): return self.net(x).squeeze(-1)

    Xt = torch.FloatTensor(X); yt = torch.FloatTensor(y)
    model = MLP(); optim = torch.optim.Adam(model.parameters(), lr=0.01); crit = nn.BCELoss()
    losses = []
    for epoch in range(30):
        optim.zero_grad(); out = model(Xt); loss = crit(out, yt)
        loss.backward(); optim.step(); losses.append(loss.item())
    with torch.no_grad():
        probs = model(Xt).numpy()
        preds_dl = (probs > 0.5).astype(int)
        acc = (preds_dl == y).mean()
    print(f"  MLP accuracy: {acc:.3f}  Final loss: {losses[-1]:.4f}")
    dl_available = True
except ImportError:
    print("  PyTorch not available — using sklearn MLP")
    from sklearn.neural_network import MLPClassifier
    mlp = MLPClassifier(hidden_layer_sizes=(32,), max_iter=200, random_state=42)
    mlp.fit(X, y); acc = mlp.score(X, y)
    print(f"  Sklearn MLP accuracy: {acc:.3f}")
    dl_available = False

# ─── MODULE 3: NLP (Week 4) ──────────────────────────────────────────────────
print("\n[MODULE 3] NLP — TF-IDF + Text Classification")
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

DOCS = [
    ("AI achieves breakthrough in protein structure prediction.", "science"),
    ("Federal Reserve raises interest rates by 75 basis points.", "finance"),
    ("Champions League final draws record global viewership.", "sports"),
    ("SpaceX successfully launches Starship on orbital mission.", "science"),
    ("Inflation hits 40-year high as energy prices surge.", "finance"),
    ("Wimbledon champion defeats top seed in straight sets.", "sports"),
]
texts = [d for d,_ in DOCS]; labels=[l for _,l in DOCS]
vec   = TfidfVectorizer(max_features=100); Xnlp = vec.fit_transform(texts)
nb    = MultinomialNB(); nb.fit(Xnlp, labels)
test  = ["Scientist wins Nobel Prize for quantum physics research.",
         "Hedge fund posts 30% returns in volatile market.",
         "Sprinter wins gold medal at World Athletics Championships."]
predictions = nb.predict(vec.transform(test))
for doc, pred in zip(test, predictions):
    print(f"  [{pred:8s}] {doc[:60]}")

# ─── MODULE 4: Agent (Week 6) ────────────────────────────────────────────────
print("\n[MODULE 4] Agentic AI — Tool Registry + Planning")

class ToolRegistry:
    def __init__(self): self._t={}
    def register(self,n,f): self._t[n]=f
    def call(self,n,**kw):
        try: return self._t[n](**kw)
        except Exception as e: return f"Error:{e}"
    def tools(self): return list(self._t.keys())

registry = ToolRegistry()
registry.register("classify_text",  lambda text: nb.predict(vec.transform([text]))[0])
registry.register("sentiment",      lambda text: "POS" if any(w in text.lower() for w in ["win","achieve","record","success"]) else "NEG")
registry.register("summarize",      lambda texts: ". ".join(t[:40] for t in texts[:2]))
registry.register("count_entities", lambda text: len(re.findall(r"\b[A-Z][a-z]+\b", text)))

agent_queries = [
    ("classify_text", {"text": "Athlete sets new Olympic world record"}),
    ("sentiment",     {"text": "Stock market achieves record high gains"}),
    ("count_entities",{"text": "Apple and Google announced a new deal in London"}),
]
for tool, args in agent_queries:
    result = registry.call(tool, **args)
    print(f"  Tool:{tool:<18}  Result: {result}")

# ─── MODULE 5: Evaluation Dashboard ──────────────────────────────────────────
print("\n[MODULE 5] Evaluation & Visualization")
fig, axes = plt.subplots(1, 3, figsize=(13, 4))

# ML CV scores
axes[0].bar(["LR CV acc"], [cv.mean()], yerr=[cv.std()], color="steelblue", capsize=6)
axes[0].set_ylim(0, 1); axes[0].set_title("ML: CV Accuracy"); axes[0].grid(axis="y", alpha=0.3)
axes[0].text(0, cv.mean()+0.02, f"{cv.mean():.2f}", ha="center")

# NLP confusion summary
pred_all   = nb.predict(Xnlp)
cat_correct= Counter(l for l,p in zip(labels,pred_all) if l==p)
cat_wrong  = Counter(l for l,p in zip(labels,pred_all) if l!=p)
cats=sorted(set(labels))
x=np.arange(len(cats))
axes[1].bar(x-0.2,[cat_correct.get(c,0) for c in cats],0.4,label="Correct",color="seagreen")
axes[1].bar(x+0.2,[cat_wrong.get(c,0) for c in cats],0.4,label="Wrong",color="tomato")
axes[1].set_xticks(x); axes[1].set_xticklabels(cats,fontsize=8)
axes[1].set_title("NLP: Per-Class Accuracy"); axes[1].legend(fontsize=8); axes[1].grid(axis="y",alpha=0.3)

# Agent tool usage
tool_calls = [t for t,_ in agent_queries]
tc_count   = Counter(tool_calls)
axes[2].pie(tc_count.values(), labels=tc_count.keys(), autopct="%d%%", startangle=90)
axes[2].set_title("Agent Tool Usage")

plt.suptitle("42-Day Python AI Curriculum — Integration Demo", fontsize=11, fontweight="bold")
plt.tight_layout(); plt.savefig("full_demo.png", dpi=90); plt.close()
print("  Saved full_demo.png")

print("""
=== CONGRATULATIONS ===
You have completed the 42-day Python AI/ML curriculum!

Week 1-2: Data science foundations (NumPy, Pandas, Sklearn)
Week 3:   Deep learning with PyTorch (MLP, CNN, RNN, LSTM)
Week 4:   NLP (sentiment, NER, classification, generation)
Week 5:   Advanced DL (autoencoders, GANs, attention, transformers)
Week 6:   Agentic AI (planning, tool use, multi-agent, guardrails)

This demo combined all modules into a single runnable pipeline.
You are now ready to build real-world AI systems.
""")

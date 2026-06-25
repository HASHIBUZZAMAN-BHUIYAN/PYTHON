"""
Project: Agent Benchmark Suite
Teaches: systematic evaluation of multiple agents across categories,
         accuracy/speed/safety metrics, comparison charts.
~20 MB RAM, ~1s on CPU
"""
import time, random
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from dataclasses import dataclass

@dataclass
class TestCase:
    query:    str
    expected: str
    category: str
    weight:   float = 1.0

# ─── Agent implementations ────────────────────────────────────────────────────
KNOWLEDGE_BASE = {
    "capital of france": "paris", "capital of germany": "berlin",
    "capital of japan": "tokyo", "capital of brazil": "brasilia",
    "2+2": "4", "3*3": "9", "square root of 16": "4",
    "python": "high-level language", "machine learning": "learn from data",
    "deep learning": "neural networks", "neural network": "layers of neurons",
}

def keyword_agent(q):
    q_lower = q.lower()
    for k, v in KNOWLEDGE_BASE.items():
        if k in q_lower:
            time.sleep(random.uniform(0.0001, 0.001))
            return v
    return "unknown"

def enhanced_agent(q):
    q_lower = q.lower()
    for k, v in KNOWLEDGE_BASE.items():
        if k in q_lower:
            time.sleep(random.uniform(0.001, 0.005))
            return f"Answer: {v} (high confidence)"
    # Partial match
    for k, v in KNOWLEDGE_BASE.items():
        if any(w in q_lower for w in k.split() if len(w)>3):
            return f"Partial: {v}"
    return "I don't know."

def slow_agent(q):
    time.sleep(random.uniform(0.01, 0.02))
    return keyword_agent(q)

def random_agent(q):
    answers = list(KNOWLEDGE_BASE.values()) + ["unknown", "I don't know"]
    return random.choice(answers)

AGENTS = {
    "Keyword":  keyword_agent,
    "Enhanced": enhanced_agent,
    "Slow":     slow_agent,
    "Random":   random_agent,
}

SUITE = [
    TestCase("What is the capital of France?",      "paris",            "geography",  1.0),
    TestCase("What is the capital of Germany?",     "berlin",           "geography",  1.0),
    TestCase("What is the capital of Japan?",       "tokyo",            "geography",  1.0),
    TestCase("What is 2+2?",                        "4",                "math",       1.0),
    TestCase("Calculate 3*3",                       "9",                "math",       1.0),
    TestCase("Square root of 16",                   "4",                "math",       1.0),
    TestCase("Describe Python language",            "high-level",       "tech",       1.0),
    TestCase("What is machine learning?",           "learn from data",  "tech",       1.5),
    TestCase("Explain deep learning",               "neural networks",  "tech",       1.5),
    TestCase("Who is the best programmer?",         "unknown",          "unknown",    0.5),
]

print("=== Agent Benchmark Suite ===\n")
results = {}
for agent_name, agent_fn in AGENTS.items():
    random.seed(42)
    scores = []; latencies = []; correct = 0; total_weight = 0
    for tc in SUITE:
        t0 = time.time()
        answer = agent_fn(tc.query)
        lat    = (time.time() - t0) * 1000
        latencies.append(lat)
        match = tc.expected.lower() in answer.lower()
        scores.append(match * tc.weight)
        if match: correct += 1
        total_weight += tc.weight
    weighted_acc = sum(scores) / total_weight
    avg_lat = np.mean(latencies)
    p95_lat = np.percentile(latencies, 95)
    results[agent_name] = {"accuracy": weighted_acc, "correct": correct,
                           "avg_ms": avg_lat, "p95_ms": p95_lat, "latencies": latencies}
    print(f"  {agent_name:>10}: acc={weighted_acc:.1%}  correct={correct}/{len(SUITE)}  avg={avg_lat:.2f}ms  p95={p95_lat:.2f}ms")

# ─── Visualization ─────────────────────────────────────────────────────────
names  = list(results.keys())
acc    = [results[n]["accuracy"] for n in names]
avg_ms = [results[n]["avg_ms"]   for n in names]

fig, axes = plt.subplots(1, 3, figsize=(13, 4))

# Accuracy bar
axes[0].bar(names, [a*100 for a in acc], color=["steelblue","seagreen","tomato","orange"])
axes[0].set_title("Accuracy (%)"); axes[0].set_ylim(0,100); axes[0].grid(axis="y",alpha=0.3)
for i, v in enumerate(acc): axes[0].text(i, v*100+1, f"{v:.0%}", ha="center", fontsize=9)

# Latency bar
axes[1].bar(names, avg_ms, color=["steelblue","seagreen","tomato","orange"])
axes[1].set_title("Avg Latency (ms)"); axes[1].grid(axis="y",alpha=0.3)

# Scatter: accuracy vs latency
scatter_colors = ["steelblue","seagreen","tomato","orange"]
for i, n in enumerate(names):
    axes[2].scatter(avg_ms[i], acc[i]*100, color=scatter_colors[i], s=100, label=n, zorder=3)
axes[2].set_xlabel("Avg Latency (ms)"); axes[2].set_ylabel("Accuracy (%)")
axes[2].set_title("Accuracy vs Speed Tradeoff"); axes[2].legend(fontsize=8); axes[2].grid(alpha=0.3)

plt.suptitle("Agent Benchmark Results", fontsize=11)
plt.tight_layout(); plt.savefig("benchmark.png", dpi=85); plt.close()
print("\nSaved benchmark.png")

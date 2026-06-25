"""
Project: Research Question Decomposition Agent
Teaches: breaking a research question into a multi-step investigation plan,
         simulating search/analysis/synthesis with mock tools.
~10 MB RAM, <1s on CPU
"""
from collections import OrderedDict

# ─── Mock research tools ──────────────────────────────────────────────────────
def search(query):
    db = {
        "transformer architecture":    "Vaswani 2017 introduced multi-head attention, positional encoding, encoder-decoder.",
        "bert pretraining":            "Devlin 2019 BERT: Bidirectional encoder, masked LM + next sentence prediction.",
        "gpt language model":          "OpenAI GPT: autoregressive transformer, unidirectional, used for generation.",
        "attention mechanism history": "Bahdanau 2014: first neural attention for machine translation alignment.",
        "fine-tuning transformers":    "Task-specific head on top of pretrained encoder; learning rate 2e-5, 3 epochs.",
        "transformer applications":    "NLP (translation, QA, summarization), vision (ViT), code (Codex), speech.",
    }
    for k,v in db.items():
        if any(w in query.lower() for w in k.split()): return v
    return f"No direct result found for '{query}'. Try related terms."

def analyze(text):
    words = text.split(); key_terms = [w for w in words if len(w)>5 and w.isalpha()]
    return f"Key terms: {key_terms[:5]}. Main claim in first 50 chars: '{text[:50]}'"

def compare(a, b):
    return f"Comparing '{a[:40]}' vs '{b[:40]}': Both use neural architectures; differ in directionality and objective."

def synthesize(findings):
    return ("Synthesis: Transformers (Vaswani 2017) enabled parallel sequence processing via attention. "
            f"BERT ({len(findings)} sources analyzed) extended this with bidirectional pretraining. "
            "Modern LLMs scale these ideas to billions of parameters.")

# ─── Plan-then-Execute research agent ─────────────────────────────────────────
class ResearchAgent:
    PLAN_TEMPLATE = [
        ("search",    "Find foundational transformer paper",     "transformer architecture"),
        ("search",    "Find BERT paper details",                 "bert pretraining"),
        ("search",    "Find GPT approach",                       "gpt language model"),
        ("search",    "Find attention mechanism origins",        "attention mechanism history"),
        ("analyze",   "Analyze transformer finding",             "__result_0__"),
        ("analyze",   "Analyze BERT finding",                    "__result_1__"),
        ("compare",   "Compare BERT and GPT approaches",         "__result_1__||__result_2__"),
        ("search",    "Find downstream fine-tuning methods",     "fine-tuning transformers"),
        ("synthesize","Synthesize all findings into conclusion",  "__all_results__"),
    ]

    def __init__(self):
        self.results = OrderedDict()
        self.tools   = {"search":search,"analyze":analyze,"compare":compare,"synthesize":synthesize}

    def resolve_arg(self, arg):
        if arg.startswith("__result_"):
            idx = int(arg.strip("_").split("_")[-1])
            return list(self.results.values())[idx] if idx < len(self.results) else ""
        if "||" in arg:
            parts = arg.split("||")
            return (self.resolve_arg(parts[0]), self.resolve_arg(parts[1]))
        if arg == "__all_results__":
            return list(self.results.values())
        return arg

    def run(self, question):
        print(f"Research Question: {question}\n")
        print("=== Generating Plan ===")
        for i, (tool, desc, _) in enumerate(self.PLAN_TEMPLATE):
            print(f"  Step {i+1}: [{tool.upper()}] {desc}")

        print("\n=== Executing Plan ===")
        for i, (tool_name, desc, arg_template) in enumerate(self.PLAN_TEMPLATE):
            arg = self.resolve_arg(arg_template)
            fn  = self.tools[tool_name]
            if isinstance(arg, tuple): result = fn(*arg)
            elif isinstance(arg, list): result = fn(arg)
            else: result = fn(arg)
            self.results[f"step_{i}_{tool_name}"] = result
            print(f"\n  Step {i+1} [{tool_name}]: {desc}")
            print(f"    Result: {str(result)[:100]}")

        return self.results

agent = ResearchAgent()
results = agent.run("How do Transformer models work and what made BERT/GPT successful?")

print("\n\n=== RESEARCH CONCLUSION ===")
print(results[list(results.keys())[-1]])
print(f"\nTotal research steps: {len(results)}")

# AGENTIC_AI Reference — LLM Tool-Calling Agent Template
# Runs in mock mode by default.
# Set ANTHROPIC_API_KEY for real Claude API calls.
# ~10 MB RAM, <1s on CPU (mock mode)

import os, re, json, math, datetime

# ─── TOOL REGISTRY ───────────────────────────────────────────────────────────
TOOLS = {}

def register_tool(name, description):
    """Decorator to register a function as a tool."""
    def decorator(fn):
        TOOLS[name] = {"fn": fn, "description": description}
        return fn
    return decorator

@register_tool("calculator", "Evaluate a Python math expression. Example: calculator('2**10 + sqrt(9)')")
def calculator(expr: str) -> str:
    try:
        allowed = {k: getattr(math, k) for k in dir(math) if not k.startswith("_")}
        return f"Result: {eval(expr, {'__builtins__': {}}, allowed)}"
    except Exception as e:
        return f"Error: {e}"

@register_tool("get_date", "Get the current date and time.")
def get_date() -> str:
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

@register_tool("lookup", "Look up a topic in the knowledge base. Example: lookup('machine learning')")
def lookup(topic: str) -> str:
    KB = {
        "python":         "Python 3.12: high-level, interpreted. Great for ML/AI.",
        "machine learning":"ML: systems that learn from data. Algorithms: LinearReg, DT, SVM, NN.",
        "deep learning":  "DL: neural networks with many layers. Frameworks: PyTorch, TensorFlow.",
        "transformer":    "Transformer (2017): self-attention architecture. Foundation of LLMs.",
        "pid controller": "PID: Proportional+Integral+Derivative control. Used in robotics/automation.",
    }
    t = topic.lower().strip()
    for k, v in KB.items():
        if t in k or k in t: return v
    return f"No result for '{topic}'."

# ─── ADD YOUR OWN TOOLS HERE ──────────────────────────────────────────────────
# @register_tool("my_tool", "Description of my_tool.")
# def my_tool(arg: str) -> str:
#     return f"Result of my_tool({arg})"

# ─── TOOL SCHEMA DESCRIPTION ─────────────────────────────────────────────────
def build_tool_schema():
    lines = ["Available tools:"]
    for name, info in TOOLS.items():
        lines.append(f"  {name}({name}_arg) — {info['description']}")
    lines += [
        "",
        "Respond in EXACTLY one of these formats:",
        "  THOUGHT: <reasoning>",
        "  ACTION: tool_name(argument)",
        "    — or —",
        "  THOUGHT: <reasoning>",
        "  FINAL_ANSWER: <your final answer>",
    ]
    return "\n".join(lines)

TOOL_SCHEMA = build_tool_schema()

# ─── MOCK LLM ────────────────────────────────────────────────────────────────
class MockLLM:
    """Deterministic mock for testing without API key."""
    def __init__(self):
        self._step = 0

    def complete(self, messages: list) -> str:
        self._step += 1
        full = " ".join(m.get("content","") for m in messages).lower()

        if "calculator" in full and "observe" in full:
            obs = next((m["content"] for m in reversed(messages) if "OBSERVE" in m.get("content","")), "")
            return f"THOUGHT: Calculation done.\nFINAL_ANSWER: {obs.replace('OBSERVE:','').strip()}"
        if "calculate" in full or re.search(r"\d+\s*[\+\-\*\/\^]\s*\d+", full):
            m = re.search(r"([\d\s\+\-\*\/\.\^\(\)]+)", full)
            expr = m.group(1).strip() if m else "1+1"
            return f"THOUGHT: I need to calculate this.\nACTION: calculator({expr})"
        if "date" in full or "time" in full:
            if "observe" in full:
                obs = next((m["content"] for m in reversed(messages) if "OBSERVE" in m.get("content","")), "")
                return f"THOUGHT: Got the date.\nFINAL_ANSWER: {obs.replace('OBSERVE:','').strip()}"
            return "THOUGHT: I need the current date.\nACTION: get_date()"
        if "look" in full or "what is" in full or "tell me" in full:
            if "observe" in full:
                obs = next((m["content"] for m in reversed(messages) if "OBSERVE" in m.get("content","")), "")
                return f"THOUGHT: Found information.\nFINAL_ANSWER: {obs.replace('OBSERVE:','').strip()}"
            m = re.search(r"(?:what is|about|tell me about)\s+([a-z][a-z\s]+)", full)
            topic = m.group(1).strip() if m else "python"
            return f"THOUGHT: I should look this up.\nACTION: lookup({topic})"
        return f"THOUGHT: I can answer directly.\nFINAL_ANSWER: Response to: {messages[-1].get('content','')[:60]}"

# ─── ANTHROPIC LLM (optional) ────────────────────────────────────────────────
class AnthropicLLM:
    def __init__(self):
        try:
            import anthropic
            self.client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
            self.model  = "claude-haiku-4-5-20251001"
            self.available = True
            print(f"Using Anthropic API: {self.model}")
        except Exception as e:
            self.available = False

    def complete(self, messages: list) -> str:
        try:
            resp = self.client.messages.create(
                model=self.model, max_tokens=512,
                system=f"You are a helpful AI agent.\n{TOOL_SCHEMA}",
                messages=messages,
            )
            return resp.content[0].text
        except Exception as e:
            return f"THOUGHT: API error: {e}\nFINAL_ANSWER: Error calling API."

# ─── REACT AGENT ─────────────────────────────────────────────────────────────
class ReActAgent:
    def __init__(self, llm=None, max_steps=6, verbose=True):
        if llm is None:
            api_key = os.environ.get("ANTHROPIC_API_KEY")
            if api_key:
                real = AnthropicLLM()
                llm = real if real.available else MockLLM()
            else:
                llm = MockLLM()
                if verbose: print("No API key — using mock LLM")
        self.llm       = llm
        self.max_steps = max_steps
        self.verbose   = verbose

    def run(self, query: str) -> str:
        if self.verbose: print(f"\nQuery: {query}")
        messages = [{"role":"user","content":f"{TOOL_SCHEMA}\n\nTask: {query}"}]
        for step in range(self.max_steps):
            response = self.llm.complete(messages)
            am = re.search(r"ACTION:\s*(\w+)\(([^)]*)\)", response)
            fm = re.search(r"FINAL_ANSWER:\s*(.+)", response, re.DOTALL)
            th = re.search(r"THOUGHT:\s*(.+?)(?:\n(?:ACTION|FINAL_ANSWER)|$)", response, re.DOTALL)
            if self.verbose and th:
                print(f"  [Step {step+1}] THINK: {th.group(1).strip()[:80]}")
            if fm:
                ans = fm.group(1).strip()
                if self.verbose: print(f"  [Step {step+1}] ANSWER: {ans[:120]}")
                return ans
            if am:
                tool_name = am.group(1)
                tool_arg  = am.group(2).strip("'\"")
                if self.verbose: print(f"  [Step {step+1}] ACTION: {tool_name}({tool_arg!r})")
                if tool_name in TOOLS:
                    obs = TOOLS[tool_name]["fn"](tool_arg) if tool_arg else TOOLS[tool_name]["fn"]()
                else:
                    obs = f"Unknown tool: {tool_name}"
                if self.verbose: print(f"  [Step {step+1}] OBSERVE: {str(obs)[:80]}")
                messages += [
                    {"role":"assistant","content": response},
                    {"role":"user",     "content": f"OBSERVE: {obs}\nContinue."},
                ]
            else:
                if self.verbose: print(f"  [Step {step+1}] (unexpected): {response[:60]}")
                break
        return "max steps reached"


# ─── DEMO ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    agent = ReActAgent(verbose=True)
    queries = [
        "What is 2 ** 10 + 100?",
        "What is today's date?",
        "Tell me about the transformer architecture.",
    ]
    for q in queries:
        agent.run(q)

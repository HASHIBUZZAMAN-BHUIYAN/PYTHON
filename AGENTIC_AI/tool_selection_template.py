# AGENTIC_AI Reference — Tool Selection & Chaining Template
# Dynamic tool selection, chaining, retry on failure, logging.
# ~10 MB RAM, <1s on CPU

import time
import json
import re
from functools import wraps
from typing import Callable, Any

# ─── 1. TOOL REGISTRY ─────────────────────────────────────────────────────────
class ToolRegistry:
    """
    Central registry for all agent tools.
    Supports: register, call, list, describe.
    """
    def __init__(self):
        self._tools: dict = {}   # name → {"fn", "desc", "schema"}

    def register(self, name: str, description: str, schema: dict = None):
        """Decorator to register a function as a tool."""
        def decorator(fn: Callable):
            self._tools[name] = {
                "fn":     fn,
                "desc":   description,
                "schema": schema or {},
            }
            return fn
        return decorator

    def call(self, name: str, *args, **kwargs) -> Any:
        if name not in self._tools:
            raise KeyError(f"Unknown tool: {name}")
        return self._tools[name]["fn"](*args, **kwargs)

    def list_tools(self) -> list:
        return list(self._tools.keys())

    def describe(self) -> str:
        lines = ["Available tools:"]
        for name, info in self._tools.items():
            lines.append(f"  {name}: {info['desc']}")
        return "\n".join(lines)

    def __contains__(self, name): return name in self._tools

# ─── 2. LOGGING DECORATOR ─────────────────────────────────────────────────────
class ToolLogger:
    """Log all tool calls with timing."""
    def __init__(self):
        self.calls: list = []

    def wrap(self, tool_fn: Callable, tool_name: str) -> Callable:
        logger = self
        @wraps(tool_fn)
        def wrapper(*args, **kwargs):
            start = time.perf_counter()
            error = None; result = None
            try:
                result = tool_fn(*args, **kwargs)
            except Exception as e:
                error = str(e)
            elapsed_ms = (time.perf_counter() - start) * 1000
            logger.calls.append({
                "tool":    tool_name,
                "args":    str(args)[:60],
                "result":  str(result)[:60] if result else None,
                "error":   error,
                "ms":      round(elapsed_ms, 2),
            })
            if error: raise RuntimeError(error)
            return result
        return wrapper

    def summary(self):
        from collections import Counter
        counts  = Counter(c["tool"] for c in self.calls)
        avg_ms  = {t: sum(c["ms"] for c in self.calls if c["tool"]==t) /
                      max(counts[t], 1) for t in counts}
        errors  = [c for c in self.calls if c["error"]]
        print(f"  Total calls: {len(self.calls)}")
        print(f"  Per-tool counts: {dict(counts)}")
        print(f"  Avg latency (ms): {avg_ms}")
        print(f"  Errors: {len(errors)}")

    def print_trace(self):
        print(f"{'Tool':<18} {'Args':<25} {'Result':<30} {'ms':>6}")
        print("-"*82)
        for c in self.calls:
            print(f"  {c['tool']:<16} {c['args']:<23} "
                  f"{(c['result'] or c['error'] or ''):<28} {c['ms']:>6.1f}")

# ─── 3. RETRY WRAPPER ─────────────────────────────────────────────────────────
def with_retry(tool_fn: Callable, max_retries=3, backoff_base=0.1):
    """Wrap a tool call with exponential backoff retry."""
    @wraps(tool_fn)
    def wrapper(*args, **kwargs):
        last_err = None
        for attempt in range(1, max_retries + 1):
            try:
                return tool_fn(*args, **kwargs)
            except Exception as e:
                last_err = e
                if attempt < max_retries:
                    wait = backoff_base * (2 ** (attempt - 1))
                    time.sleep(wait)
        raise RuntimeError(f"Tool failed after {max_retries} retries: {last_err}")
    return wrapper

# ─── 4. TOOL CHAIN ────────────────────────────────────────────────────────────
class ToolChain:
    """Execute a sequence of tool calls where each output feeds the next."""
    def __init__(self, registry: ToolRegistry, logger: ToolLogger = None):
        self.registry = registry
        self.logger   = logger

    def run_chain(self, steps: list, verbose=True) -> Any:
        """
        steps: list of (tool_name, arg_template_fn)
        arg_template_fn receives previous result and returns args dict/str.
        """
        result = None
        for tool_name, arg_fn in steps:
            args = arg_fn(result) if result is not None else arg_fn(None)
            if verbose: print(f"  CHAIN: {tool_name}({str(args)[:40]})")
            fn = self.registry._tools[tool_name]["fn"]
            if self.logger:
                fn = self.logger.wrap(fn, tool_name)
            if isinstance(args, dict):
                result = fn(**args)
            elif isinstance(args, str):
                result = fn(args)
            else:
                result = fn()
            if verbose: print(f"    -> {str(result)[:60]}")
        return result

# ─── 5. TOOL SELECTOR (keyword-based) ────────────────────────────────────────
class ToolSelector:
    """
    Select the most appropriate tool for a query using keyword matching.
    """
    def __init__(self, tool_keywords: dict):
        """tool_keywords: {tool_name: [keyword, ...]}"""
        self.keywords = tool_keywords

    def select(self, query: str) -> str | None:
        q_lower = query.lower()
        scores  = {name: sum(1 for kw in kws if kw in q_lower)
                   for name, kws in self.keywords.items()}
        best = max(scores, key=scores.get)
        return best if scores[best] > 0 else None

# ─── DEMO ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import math

    registry = ToolRegistry()
    logger   = ToolLogger()

    @registry.register("calculator", "Evaluate math expression")
    def calculator(expr):
        allowed = {k: getattr(math,k) for k in dir(math) if not k.startswith("_")}
        return eval(expr, {"__builtins__":{}}, allowed)

    @registry.register("get_weather", "Get mock weather for city")
    def get_weather(city):
        return json.dumps({"city":city,"temp":14,"condition":"cloudy"})

    @registry.register("convert_units", "Convert units (e.g. '14 celsius to fahrenheit')")
    def convert_units(expr):
        m = re.match(r"([\d.]+)\s*celsius\s+to\s+fahrenheit", expr)
        if m: return f"{float(m.group(1))*9/5+32:.1f}°F"
        return f"cannot parse: {expr}"

    print("=== Tool Registry ===")
    print(registry.describe())

    print("\n=== Tool Chain: weather -> convert ===")
    chain = ToolChain(registry, logger)
    result = chain.run_chain([
        ("get_weather",   lambda _: "London"),
        ("convert_units", lambda r: f"{json.loads(r)['temp']} celsius to fahrenheit"),
    ])
    print(f"  Final: {result}")

    print("\n=== Tool Selector ===")
    selector = ToolSelector({
        "calculator":   ["calculate","compute","what is","math"],
        "get_weather":  ["weather","temperature","forecast","rain"],
        "convert_units":["convert","celsius","fahrenheit","kg","miles"],
    })
    queries = ["What is 2**10?", "What's the weather in Tokyo?", "Convert 100 km to miles"]
    for q in queries:
        tool = selector.select(q)
        print(f"  '{q[:35]}' -> {tool}")

    print("\n=== Call Log ===")
    logger.print_trace()
    print(); logger.summary()

# Advanced Day 38 — Tool-Using Agents II
# ~20 MB RAM, <1s on CPU

print("""
=== Tool-Using Agents II — Day 38 ===

Day 20 introduced basic tool calling. Today we add:
  1. Schema validation — typed tool signatures with auto-conversion
  2. Retry with backoff — transient failure recovery
  3. Fallback chains   — if tool A fails, try tool B
  4. Observability     — structured call logs, timing, trace IDs
  5. Conditional chains — if/else branching based on tool output
""")

import time, uuid, functools
from typing import Callable, Any

# ─── 1. TOOL REGISTRY WITH SCHEMA ────────────────────────────────────────────
print("=== 1. Typed Tool Registry ===")

class ToolError(Exception): pass

class Tool:
    def __init__(self, name, fn, description, param_types=None, required=None):
        self.name        = name
        self.fn          = fn
        self.description = description
        self.param_types = param_types or {}
        self.required    = required or list(param_types.keys())

    def __call__(self, **kwargs):
        for p in self.required:
            if p not in kwargs: raise ToolError(f"Missing required param: {p}")
        cast = {}
        for k, v in kwargs.items():
            t = self.param_types.get(k, str)
            try: cast[k] = t(v)
            except (ValueError, TypeError) as e:
                raise ToolError(f"Param '{k}' cannot be cast to {t.__name__}: {e}")
        return self.fn(**cast)

class ToolRegistry:
    def __init__(self):
        self._tools: dict[str, Tool] = {}

    def register(self, name, description, param_types=None, required=None):
        def decorator(fn):
            self._tools[name] = Tool(name, fn, description, param_types or {}, required)
            return fn
        return decorator

    def call(self, name, **kwargs):
        if name not in self._tools: raise ToolError(f"Unknown tool: {name}")
        return self._tools[name](**kwargs)

    def describe(self):
        for name, tool in self._tools.items():
            print(f"  {name}: {tool.description}  params={list(tool.param_types.keys())}")

registry = ToolRegistry()

@registry.register("add", "Add two numbers", {"a":float,"b":float})
def add(a,b): return a + b

@registry.register("divide", "Divide a by b", {"a":float,"b":float})
def divide(a,b):
    if abs(b) < 1e-10: raise ToolError("Division by zero")
    return a / b

@registry.register("search", "Search knowledge base", {"query":str})
def search(query):
    KB={"python":"A high-level programming language","ml":"Machine learning is AI subset"}
    for k,v in KB.items():
        if k in query.lower(): return v
    return "No results found."

registry.describe()
print(f"\n  add(3, 4.5) = {registry.call('add', a='3', b='4.5')}")
print(f"  search('python') = {registry.call('search', query='python language')}")

# ─── 2. RETRY WITH BACKOFF ────────────────────────────────────────────────────
print("\n=== 2. Retry with Exponential Backoff ===")

class FlakeyTool:
    def __init__(self, fail_first_n=2):
        self.calls = 0; self.fail_n = fail_first_n
    def __call__(self, x):
        self.calls += 1
        if self.calls <= self.fail_n: raise ToolError(f"Transient failure #{self.calls}")
        return f"Success on attempt {self.calls}: result={x*2}"

def with_retry(fn, max_retries=3, backoff_base=0.1):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        for attempt in range(max_retries+1):
            try: return fn(*args, **kwargs)
            except ToolError as e:
                if attempt == max_retries: raise
                delay = backoff_base * (2**attempt)
                print(f"  Retry {attempt+1}: {e} → waiting {delay:.2f}s")
                time.sleep(delay)
    return wrapper

flakey = FlakeyTool(fail_first_n=2)
robust = with_retry(flakey, max_retries=3, backoff_base=0.05)
result = robust(x=42)
print(f"  Final result: {result}")

# ─── 3. FALLBACK CHAIN ────────────────────────────────────────────────────────
print("\n=== 3. Fallback Tool Chain ===")

def primary_weather_api(city):
    if city == "London": return f"London: 15°C, cloudy"
    raise ToolError("City not in primary DB")

def backup_weather_api(city):
    return f"{city}: ~18°C, data from backup source"

def with_fallback(primary, fallback):
    def wrapper(*args, **kwargs):
        try:
            return primary(*args, **kwargs), "primary"
        except ToolError:
            return fallback(*args, **kwargs), "fallback"
    return wrapper

weather = with_fallback(primary_weather_api, backup_weather_api)
for city in ["London", "Tokyo", "Paris"]:
    result, source = weather(city)
    print(f"  {city}: {result}  [source={source}]")

# ─── 4. STRUCTURED CALL LOG ───────────────────────────────────────────────────
print("\n=== 4. Structured Observability ===")

class ObservableRegistry(ToolRegistry):
    def __init__(self):
        super().__init__(); self.log = []

    def call(self, name, trace_id=None, **kwargs):
        tid   = trace_id or str(uuid.uuid4())[:8]
        t0    = time.time()
        entry = {"trace_id":tid,"tool":name,"kwargs":kwargs,"status":"pending","ms":0}
        try:
            result = super().call(name, **kwargs)
            entry.update({"status":"ok","result":str(result)[:60],"ms":round((time.time()-t0)*1000,1)})
            return result
        except ToolError as e:
            entry.update({"status":"error","error":str(e),"ms":round((time.time()-t0)*1000,1)})
            raise
        finally:
            self.log.append(entry)

    def print_trace(self):
        for e in self.log:
            status = "✓" if e["status"]=="ok" else "✗"
            print(f"  {status} [{e['trace_id']}] {e['tool']}({e['kwargs']}) → {e.get('result',e.get('error',''))} ({e['ms']}ms)")

obs = ObservableRegistry()
obs._tools = registry._tools  # reuse existing tools
obs.call("add", a=10, b=20)
obs.call("search", query="ml topics")
try: obs.call("divide", a=1, b=0)
except ToolError: pass
obs.print_trace()

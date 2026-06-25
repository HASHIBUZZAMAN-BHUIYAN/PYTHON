# Advanced Day 38 — Solutions
import time, threading, functools
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict, deque

class ToolError(Exception): pass

# Ex 1
print("=== Ex 1: Tool Timeout ===")
def with_timeout(fn, timeout_sec):
    @functools.wraps(fn)
    def wrapper(*a, **kw):
        result=[None]; exc=[None]
        def run():
            try: result[0]=fn(*a,**kw)
            except Exception as e: exc[0]=e
        t=threading.Thread(target=run); t.start(); t.join(timeout_sec)
        if t.is_alive(): raise ToolError(f"Timeout after {timeout_sec}s")
        if exc[0]: raise exc[0]
        return result[0]
    return wrapper

def slow_tool(x): time.sleep(0.3); return x*2
fast_timeout = with_timeout(slow_tool, 1.0)
slow_timeout = with_timeout(slow_tool, 0.1)
print(f"  fast_timeout(5): {fast_timeout(5)}")
try: slow_timeout(5)
except ToolError as e: print(f"  slow_timeout: {e}")

# Ex 2
print("\n=== Ex 2: Rate Limiter ===")
def with_rate_limit(fn, max_calls=3, window_sec=60):
    calls = deque()
    @functools.wraps(fn)
    def wrapper(*a, **kw):
        now=time.time()
        while calls and calls[0]<now-window_sec: calls.popleft()
        if len(calls)>=max_calls: raise ToolError(f"Rate limit: {max_calls} calls/{window_sec}s")
        calls.append(now); return fn(*a, **kw)
    return wrapper

def my_tool(x): return x+1
limited = with_rate_limit(my_tool, max_calls=3, window_sec=60)
for i in range(5):
    try: print(f"  Call {i+1}: {limited(i)}")
    except ToolError as e: print(f"  Call {i+1}: {e}")

# Ex 3
print("\n=== Ex 3: Tool Composition ===")
def compose(tool_a, tool_b):
    def composed(*a, **kw):
        result_a = tool_a(*a, **kw)
        return tool_b(result_a)
    composed.__name__ = f"{tool_a.__name__}|{tool_b.__name__}"
    return composed

clean  = lambda t: t.lower().strip()
shorten= lambda t: t[:30]+"..." if len(t)>30 else t
process= compose(clean, shorten)
for text in ["  Hello World This Is A Very Long String  ", "SHORT"]:
    print(f"  '{text[:30]}' → '{process(text)}'")

# Ex 4
print("\n=== Ex 4: Parallel Tool Calls ===")
def run_parallel(tasks):
    results={}
    with ThreadPoolExecutor(max_workers=len(tasks)) as ex:
        futs={ex.submit(fn,**kwargs):name for name,(fn,kwargs) in tasks.items()}
        for fut in as_completed(futs):
            name=futs[fut]
            try: results[name]={"ok":True,"result":fut.result()}
            except Exception as e: results[name]={"ok":False,"error":str(e)}
    return results

def add(a,b): time.sleep(0.05); return a+b
def mul(a,b): time.sleep(0.05); return a*b
tasks={"add":(add,{"a":3,"b":4}),"mul":(mul,{"a":3,"b":4})}
t0=time.time(); res=run_parallel(tasks); elapsed=time.time()-t0
for name,r in res.items(): print(f"  {name}: {r}")
print(f"  Parallel time: {elapsed:.3f}s (both ran concurrently)")

# Ex 5
print("\n=== Ex 5: Tool Versioning ===")
class VersionedRegistry:
    def __init__(self): self._tools={}
    def register(self,name,version,fn): self._tools[(name,version)]=fn
    def call(self,name,version,**kw):
        if (name,version) in self._tools: return self._tools[(name,version)](**kw)
        # fallback to any version
        alts=[v for (n,v) in self._tools if n==name]
        if alts: print(f"  WARNING: v{version} not found, using v{alts[-1]}"); return self._tools[(name,alts[-1])](**kw)
        raise ToolError(f"No tool '{name}' registered")

vreg=VersionedRegistry()
vreg.register("compute","1.0",lambda x: x+1)
vreg.register("compute","2.0",lambda x: x*2)
print(f"  v1.0: {vreg.call('compute','1.0',x=5)}")
print(f"  v2.0: {vreg.call('compute','2.0',x=5)}")
print(f"  v3.0 fallback: {vreg.call('compute','3.0',x=5)}")

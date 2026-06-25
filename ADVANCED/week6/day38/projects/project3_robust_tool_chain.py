"""
Project: Robust Tool Chain with Fallbacks and Observability
Teaches: building a production-quality tool chain with retries, fallbacks,
         timeouts, parallel execution, and structured observability.
~15 MB RAM, <1s on CPU
"""
import time, uuid, functools, threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict

class ToolError(Exception): pass

# ─── Decorators ───────────────────────────────────────────────────────────────
def with_retry(fn, max_retries=3, base_delay=0.05):
    @functools.wraps(fn)
    def wrapper(*a, **kw):
        for i in range(max_retries+1):
            try: return fn(*a,**kw)
            except ToolError as e:
                if i==max_retries: raise
                time.sleep(base_delay*(2**i))
    return wrapper

def with_fallback(primary, backup):
    def wrapper(*a,**kw):
        try: return primary(*a,**kw),"primary"
        except ToolError: return backup(*a,**kw),"backup"
    return wrapper

def with_timeout(fn,sec):
    @functools.wraps(fn)
    def wrapper(*a,**kw):
        r=[None]; e=[None]
        def run():
            try: r[0]=fn(*a,**kw)
            except Exception as ex: e[0]=ex
        t=threading.Thread(target=run); t.daemon=True; t.start(); t.join(sec)
        if t.is_alive(): raise ToolError(f"Timeout>{sec}s")
        if e[0]: raise e[0]
        return r[0]
    return wrapper

# ─── Observable Tool Runner ────────────────────────────────────────────────────
class ObservableRunner:
    def __init__(self): self.trace=[]

    def run(self, name, fn, *args, **kwargs):
        tid=str(uuid.uuid4())[:6]; t0=time.time()
        entry={"id":tid,"tool":name,"status":"pending","ms":0}
        try:
            result=fn(*args,**kwargs)
            entry.update({"status":"ok","result":str(result)[:60],"ms":round((time.time()-t0)*1000,1)})
            return result
        except ToolError as e:
            entry.update({"status":"error","error":str(e),"ms":round((time.time()-t0)*1000,1)})
            raise
        finally:
            self.trace.append(entry)
            sym="✓" if entry["status"]=="ok" else "✗"
            print(f"  {sym}[{tid}] {name:25s} {entry['ms']:6.1f}ms  {entry.get('result',entry.get('error',''))[:50]}")

    def summary(self):
        ok=sum(1 for e in self.trace if e["status"]=="ok")
        er=sum(1 for e in self.trace if e["status"]!="ok")
        total_ms=sum(e["ms"] for e in self.trace)
        print(f"\n  Summary: {ok} ok, {er} errors, {total_ms:.1f}ms total")

# ─── Tools ────────────────────────────────────────────────────────────────────
_call_counts=defaultdict(int)
def flaky_db(key):
    _call_counts["db"]+=1
    if _call_counts["db"]<=2: raise ToolError("DB connection timeout")
    return {"id":key,"name":f"User_{key}","score":42}

def cache_db(key): return {"id":key,"name":f"CachedUser_{key}","score":35}
def enrichment(user): return {**user,"premium":user["score"]>40,"tier":"gold" if user["score"]>40 else "silver"}
def formatter(data): return f"User {data['name']} | Tier:{data['tier']} | Premium:{data['premium']}"
def slow_analytics(user): time.sleep(0.06); return f"Analytics computed for {user['id']}"
def slow_notifications(user): time.sleep(0.06); return f"Notification sent to {user['name']}"

runner = ObservableRunner()

# ─── Build the chain ─────────────────────────────────────────────────────────
print("=== Robust Tool Chain Execution ===\n")
print(f"{'Tool':<28} {'Time':>6}  Result")
print("─"*70)

# Step 1: Fetch user (with retry + fallback)
robust_db   = with_retry(flaky_db, max_retries=3, base_delay=0.02)
db_with_back= with_fallback(robust_db, cache_db)
user, source = runner.run("fetch_user(retry+fallback)", db_with_back, key="U001")
print(f"    [source={source}]")

# Step 2: Enrich
enrich_safe = with_timeout(enrichment, sec=1.0)
enriched    = runner.run("enrich_user(timeout=1s)", enrich_safe, user=user)

# Step 3: Format
formatted   = runner.run("format_output", formatter, data=enriched)

# Step 4: Parallel async tasks
print("\n  Parallel tasks (analytics + notifications):")
with ThreadPoolExecutor(max_workers=2) as ex:
    futs = {ex.submit(slow_analytics,   enriched): "analytics",
            ex.submit(slow_notifications,enriched): "notifications"}
    for fut in as_completed(futs):
        runner.run(futs[fut], lambda f=fut: f.result())

print(f"\n  Final output: {formatted}")
runner.summary()

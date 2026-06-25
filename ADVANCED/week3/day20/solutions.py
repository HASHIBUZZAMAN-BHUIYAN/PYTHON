# Advanced Day 20 — Solutions
import os, json, re, math, time

# ─── Shared tools from lesson ─────────────────────────────────────────────────
def tool_calculator(expr):
    try:
        allowed={k:getattr(math,k) for k in dir(math) if not k.startswith("_")}
        return f"Result: {eval(expr,{'__builtins__':{}},allowed)}"
    except Exception as e: return f"Error: {e}"

def tool_get_weather(city):
    data={"london":{"temp":14,"condition":"cloudy"},"tokyo":{"temp":22,"condition":"sunny"}}
    d=data.get(city.lower(),{"temp":20,"condition":"mild"})
    return json.dumps({"city":city,**d})

# Exercise 1 — Unit converter
print("=== Exercise 1: Unit Converter Tool ===")
def tool_convert_units(expr):
    expr=expr.lower().strip()
    conversions={
        ("km","miles"):0.621371,("miles","km"):1.60934,
        ("kg","pounds"):2.20462,("pounds","kg"):0.453592,
        ("meters","feet"):3.28084,("feet","meters"):0.3048,
    }
    m=re.match(r"([\d.]+)\s*(\w+)\s+to\s+(\w+)",expr)
    if not m: return "Format: '<value> <from_unit> to <to_unit>'"
    val,fr,to=float(m.group(1)),m.group(2),m.group(3)
    if fr=="fahrenheit" and to=="celsius": return f"{(val-32)*5/9:.2f}°C"
    if fr=="celsius" and to=="fahrenheit": return f"{val*9/5+32:.2f}°F"
    factor=conversions.get((fr,to))
    if factor: return f"{val*factor:.4f} {to}"
    return f"Unknown conversion: {fr} → {to}"

print(f"  100 km to miles : {tool_convert_units('100 km to miles')}")
print(f"  32 F to C       : {tool_convert_units('32 fahrenheit to celsius')}")
print(f"  1 kg to pounds  : {tool_convert_units('1 kg to pounds')}")

# Exercise 2 — Persistent agent with memory
print("\n=== Exercise 2: Persistent Memory Agent ===")
class PersistentMockLLM:
    def __init__(self): self.history=[]
    def complete(self, messages, last_obs=""):
        user_msg=messages[-1]["content"].lower() if messages else ""
        if "what did i ask" in user_msg or "previous" in user_msg:
            if self.history:
                return f"THOUGHT: I can recall previous questions.\nFINAL_ANSWER: Your previous questions were: {'; '.join(self.history[-3:])}"
            return "THOUGHT: No previous questions.\nFINAL_ANSWER: You haven't asked anything before."
        if "weather" in user_msg:
            city=re.search(r"in\s+(\w+)",user_msg); city=city.group(1) if city else "London"
            return f"THOUGHT: Check weather.\nACTION: get_weather({city})"
        if "get_weather" in user_msg and "observe" in user_msg:
            return f"THOUGHT: Got weather.\nFINAL_ANSWER: {last_obs}"
        return f"THOUGHT: Answering directly.\nFINAL_ANSWER: Response to: {messages[-1]['content'][:60]}"

class PersistentAgent:
    def __init__(self, llm, max_steps=4):
        self.llm=llm; self.max_steps=max_steps; self.qa_history=[]

    def run(self, query):
        import re as _re
        self.llm.history.append(query)
        context=""
        if self.qa_history:
            context="Recent Q&A:\n"+"\n".join(f"Q: {q}\nA: {a}" for q,a in self.qa_history[-2:])+"\n\n"
        messages=[{"role":"user","content":f"{context}Task: {query}"}]
        last_obs=""
        for step in range(self.max_steps):
            resp=self.llm.complete(messages,last_obs)
            am=_re.search(r"ACTION:\s*(\w+)\(([^)]*)\)",resp)
            fm=_re.search(r"FINAL_ANSWER:\s*(.+)",resp,_re.DOTALL)
            if fm:
                ans=fm.group(1).strip(); self.qa_history.append((query,ans[:80]))
                print(f"  Q: {query}\n  A: {ans[:100]}"); return ans
            if am:
                fn,arg=am.group(1),am.group(2).strip("'\"")
                tools={"get_weather":tool_get_weather,"calculator":tool_calculator}
                last_obs=tools.get(fn,lambda x:f"unknown tool {fn}")(arg)
                messages+=[{"role":"assistant","content":resp},{"role":"user","content":f"OBSERVE:{last_obs}\nContinue."}]
        return "max steps"

pllm=PersistentMockLLM(); pagent=PersistentAgent(pllm)
pagent.run("What is the weather in London?")
pagent.run("What did I ask before?")

# Exercise 3 — Multi-step query
print("\n=== Exercise 3: Multi-Step Query ===")
class DebugMockLLM:
    step=0
    def complete(self, messages, last_obs=""):
        self.step+=1
        full=" ".join(m["content"] for m in messages).lower()
        if self.step==1: return "THOUGHT: Get London weather first.\nACTION: get_weather(London)"
        if "observe" in full and "temp" in last_obs and self.step==2:
            d=json.loads(last_obs); t=d.get("temp",14)
            return f"THOUGHT: Temperature={t}. Now calculate.\nACTION: calculator(({t}**2+10)/3)"
        if "result:" in last_obs: return f"THOUGHT: Done.\nFINAL_ANSWER: London temp=14°C, formula result={last_obs.split(':')[1].strip()}"
        return f"THOUGHT: final\nFINAL_ANSWER: Completed in {self.step} steps."

import re
dllm=DebugMockLLM()
class DebugAgent:
    def __init__(self, llm): self.llm=llm
    def run(self, q):
        print(f"  Query: {q}"); messages=[{"role":"user","content":q}]; last_obs=""
        for i in range(5):
            resp=self.llm.complete(messages,last_obs)
            am=re.search(r"ACTION:\s*(\w+)\(([^)]*)\)",resp)
            fm=re.search(r"FINAL_ANSWER:\s*(.+)",resp,re.DOTALL)
            print(f"  Step {i+1}: {resp[:80].strip()}...")
            if fm: print(f"  → FINAL: {fm.group(1).strip()[:100]}"); return
            if am:
                fn,arg=am.group(1),am.group(2).strip("'\"")
                tools={"get_weather":tool_get_weather,"calculator":tool_calculator}
                last_obs=tools.get(fn,lambda x:"err")(arg)
                print(f"  OBSERVE: {last_obs[:60]}")
                messages+=[{"role":"assistant","content":resp},{"role":"user","content":f"OBSERVE:{last_obs}\nContinue."}]

DebugAgent(DebugMockLLM()).run("What is the current temperature in London, and what is (temp^2 + 10) / 3?")

# Exercise 4 — Error handling
print("\n=== Exercise 4: Error Handling ===")
def tool_divide(expr):
    m=re.match(r"([\d.]+)\s*/\s*([\d.]+)",str(expr))
    if not m: return "Format: 'a / b'"
    a,b=float(m.group(1)),float(m.group(2))
    if b==0: raise ZeroDivisionError("Division by zero!")
    return f"Result: {a/b}"

def safe_tool_divide(expr):
    try: return tool_divide(expr)
    except ZeroDivisionError as e: return f"ERROR: {e} — division by zero is undefined."
    except Exception as e: return f"ERROR: {e}"

print(f"  10 / 2  = {safe_tool_divide('10 / 2')}")
print(f"  5  / 0  = {safe_tool_divide('5 / 0')}")

# Exercise 5 — Streaming mock
print("\n=== Exercise 5: Streaming Output ===")
def stream_print(text, delay=0.005):
    for char in text:
        print(char, end="", flush=True)
        time.sleep(delay)
    print()

class StreamingMockLLM:
    def complete(self, messages, last_obs=""):
        return "THOUGHT: I will answer this directly.\nFINAL_ANSWER: Streaming response complete."

class StreamingAgent:
    def __init__(self, llm): self.llm=llm
    def run(self, q):
        print(f"  Query: {q}"); messages=[{"role":"user","content":q}]
        resp=self.llm.complete(messages,"")
        print("  Streaming: ", end="")
        stream_print(resp, delay=0.003)

StreamingAgent(StreamingMockLLM()).run("Hello, what can you do?")

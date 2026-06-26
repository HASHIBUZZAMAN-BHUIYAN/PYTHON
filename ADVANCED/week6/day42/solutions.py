# Advanced Day 42 — Solutions
import re, time, random
from collections import deque

INTENTS={"greeting":["hello","hi","hey"],"farewell":["bye","goodbye"],
         "calculation":["calculate","what is","add","multiply","divide","sqrt","percent"],
         "weather":["weather","temperature","rain","forecast"],
         "search":["find","search","look up","tell me about","who is"]}
def classify_intent(text):
    t=text.lower(); sc={}
    for intent,kws in INTENTS.items():
        s=sum(1 for kw in kws if kw in t)
        if s: sc[intent]=s
    if not sc: return "unknown",0.0
    best=max(sc,key=sc.get); return best,sc[best]/sum(sc.values())

print("=== Ex 1: Slot Filling ===")
def slot_filling(text):
    text_l=text.lower(); slots={}
    if "sqrt" in text_l or "square root" in text_l:
        slots["operation"]="sqrt"
        nums=re.findall(r"\b\d+(?:\.\d+)?\b",text)
        if nums: slots["number"]=float(nums[0])
    elif "%" in text or "percent" in text_l:
        slots["operation"]="percent"
        nums=re.findall(r"\b\d+(?:\.\d+)?\b",text)
        if len(nums)>=2: slots["a"],slots["b"]=float(nums[0]),float(nums[1])
    else:
        ops=re.findall(r"\b(add|plus|subtract|minus|multiply|times|divide)\b",text_l)
        if ops: slots["operation"]=ops[0]
        nums=re.findall(r"\b\d+(?:\.\d+)?\b",text)
        if len(nums)>=2: slots["a"],slots["b"]=float(nums[0]),float(nums[1])
    return slots
for q in ["what is the square root of 81?","what is 15% of 200?","add 17 and 25"]:
    print(f"  '{q}' → {slot_filling(q)}")

print("\n=== Ex 2: Conversation Memory ===")
class ConversationMemory:
    def __init__(self,max_turns=10):
        self.turns=deque(maxlen=max_turns)
    def remember(self,user,asst): self.turns.append({"user":user,"asst":asst})
    def context(self,n=3):
        recent=list(self.turns)[-n:]
        return "\n".join(f"U:{t['user']} A:{t['asst']}" for t in recent)
    def relevant(self,query):
        qw=set(query.lower().split())
        best=None; best_score=0
        for t in self.turns:
            tw=set(t["user"].lower().split())
            score=len(qw&tw)
            if score>best_score: best_score=score; best=t
        return best
mem=ConversationMemory(max_turns=5)
mem.remember("Hello","Hi! How can I help?")
mem.remember("What is 2+2?","2+2=4")
mem.remember("Who is Einstein?","Albert Einstein was a physicist")
print(f"  Context(2):\n{mem.context(2)}")
print(f"  Relevant('who is a physicist'): {mem.relevant('who is a physicist')}")

print("\n=== Ex 3: Retry + Fallback ===")
def safe_agent(query,max_retries=2):
    intent,_=classify_intent(query)
    nums=re.findall(r"\b\d+(?:\.\d+)?\b",query)
    for attempt in range(max_retries+1):
        try:
            if intent=="calculation" and len(nums)>=2:
                a,b=float(nums[0]),float(nums[1])
                if "divide" in query.lower() and b==0: raise ZeroDivisionError
                return a+b,"primary",attempt
            raise ValueError("Can't compute")
        except ZeroDivisionError:
            return f"Cannot divide by zero","error-handled",attempt
        except ValueError:
            if attempt<max_retries: time.sleep(0.001); continue
            break
    # fallback to search
    return f"Search fallback for: {query[:40]}","fallback",max_retries
for q in ["add 5 and 3","divide 10 by 0","who invented Python?"]:
    result,source,attempts=safe_agent(q)
    print(f"  '{q[:35]}'  →  {result}  [{source}, attempts={attempts}]")

print("\n=== Ex 4: Evaluation Harness ===")
TEST_SUITE=[("Hello","greeting"),("What is 5 plus 3","calculation"),
            ("Search for Python tutorials","search"),("Bye","farewell"),
            ("Weather today","weather"),("Calculate 10 times 4","calculation"),
            ("Find Einstein","search"),("Hi there","greeting"),
            ("Goodbye everyone","farewell"),("What is rain like","weather")]
correct=0; latencies=[]
for query,expected in TEST_SUITE:
    t0=time.time(); intent,_=classify_intent(query); latencies.append((time.time()-t0)*1000)
    match=intent==expected; correct+=match
    print(f"  {'✓' if match else '✗'} '{query[:30]:<30}'  expected={expected:<12}  got={intent}")
import numpy as np
print(f"\n  Accuracy: {correct}/{len(TEST_SUITE)} = {correct/len(TEST_SUITE):.0%}")
print(f"  Latency: mean={np.mean(latencies):.3f}ms  p95={np.percentile(latencies,95):.3f}ms")

print("\n=== Ex 5: Full Integration (Non-interactive) ===")
import re as _re
PII_PATS={"email":r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+","ssn":r"\b\d{3}-\d{2}-\d{4}\b"}
TOXIC=[r"\bhow to (make|build|create)\s+(bomb|weapon|drug)"]
def guardrail(t):
    for p in TOXIC:
        if _re.search(p,t,_re.IGNORECASE): return False,"toxic"
    for k,p in PII_PATS.items():
        if _re.search(p,t): return False,f"pii:{k}"
    return True,None

memory=ConversationMemory()
tools={"add":lambda a,b:a+b,"multiply":lambda a,b:a*b}

def full_pipeline(query):
    ok,reason=guardrail(query)
    if not ok: return f"[BLOCKED:{reason}]"
    intent,conf=classify_intent(query)
    slots=slot_filling(query)
    if intent=="calculation" and "a" in slots and "b" in slots:
        op={"add":"add","plus":"add","multiply":"multiply","times":"multiply"}.get(slots.get("operation","add"),"add")
        ans=tools.get(op,lambda a,b:f"{a}+{b}")(slots["a"],slots["b"])
        resp=f"Result: {slots['a']} {slots.get('operation','+')} {slots['b']} = {ans}"
    elif intent=="greeting": resp="Hello! How can I help?"
    elif intent=="farewell": resp="Goodbye! Come back anytime."
    else: resp=f"[Mock response for intent={intent}]"
    memory.remember(query,resp)
    return resp

QUERIES=["Hello!","Add 12 and 8","How to make a bomb?","Multiply 6 and 7","My SSN is 123-45-6789","Bye"]
for q in QUERIES:
    r=full_pipeline(q); print(f"  Q: {q:<40}  A: {r}")
print(f"\n  Memory context:\n{memory.context(3)}")

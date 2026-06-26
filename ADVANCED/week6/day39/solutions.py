# Advanced Day 39 — Solutions
import random
from collections import Counter

class Agent:
    def __init__(self,name,role,fn):
        self.name=name; self.role=role; self.fn=fn
    def respond(self,ctx=""): return self.fn(ctx)

print("=== Ex 1: Star Topology ===")
class StarOrchestrator:
    def __init__(self,coordinator,workers):
        self.coord=coordinator; self.workers=workers
    def run(self,task):
        print(f"  Coordinator distributing: '{task[:50]}'")
        partials=[w.respond(task) for w in self.workers]
        for w,r in zip(self.workers,partials):
            print(f"    [{w.name}]: {r[:60]}")
        merged=self.coord.respond(" | ".join(partials))
        print(f"  Coordinator merged: {merged[:80]}")
        return merged

workers=[Agent("W1","NLP",  lambda p: f"NLP analysis: sentiment=positive"),
         Agent("W2","Stats",lambda p: f"Stats: 45 words, readability=8.2"),
         Agent("W3","SEO",  lambda p: f"SEO: 3 keywords found, density ok")]
coord=Agent("Coord","Merger",lambda p: f"MERGED REPORT: {p[:80]}")
star=StarOrchestrator(coord,workers); star.run("Analyze this product review text")

print("\n=== Ex 2: Escalation Chain ===")
class EscalationChain:
    def __init__(self,agents,threshold=0.5):
        self.agents=agents; self.thresh=threshold
    def run(self,query):
        for agent in self.agents:
            conf=random.random()
            result=agent.respond(query)
            print(f"  [{agent.name}({agent.role})] conf={conf:.2f} → {result[:50]}")
            if conf>=self.thresh: return result,agent.name
        return f"Escalated to all experts: {result}","all"

random.seed(7)
chain=EscalationChain([Agent("Junior","L1",lambda p:"Basic answer: "+p[:20]),
                       Agent("Senior","L2",lambda p:"Better answer: "+p[:25]),
                       Agent("Expert","L3",lambda p:"Expert answer: "+p[:30])])
ans,who=chain.run("What is the capital of Bhutan?"); print(f"  Final: {who} answered\n")

print("=== Ex 3: Collaborative Writing ===")
def outliner(p):  return "1. Intro 2. History 3. Applications 4. Conclusion"
def writer(p):    return "1. AI is transforming industries. 2. From Turing 1950... 3. Healthcare, finance, NLP... 4. Balanced adoption needed."
def checker(p):   return p.replace("Turing 1950","Turing 1950 [VERIFIED]").replace("Healthcare","Healthcare [VERIFIED]")
def editor(p):    lines=[l for l in p.split(".")if "[UNVERIFIED]" not in l]; return ". ".join(lines)
pipeline=[Agent("Outliner","plan",outliner),Agent("Writer","write",writer),
          Agent("Checker","check",checker),Agent("Editor","edit",editor)]
text=p="Explain AI"
for ag in pipeline:
    text=ag.respond(text); print(f"  [{ag.name}]: {text[:80]}")

print("\n=== Ex 4: Weighted Voting ===")
LEXICON={"good","great","excellent","love","amazing"}
def lex(t): return "POS" if any(w in t.lower() for w in LEXICON) else "NEG"
def punc(t): return "POS" if t.count("!")>0 else "NEG"
def length(t): return "POS" if len(t.split())>8 else "NEG"
def caps(t): return "POS" if sum(1 for c in t if c.isupper())<5 else "NEG"
def excl(t): return "POS" if t.count("?")>0 else "NEG"
weights=[0.6,0.5,0.5,0.4,0.7]; fns=[lex,punc,length,caps,excl]
for text in ["This is a great product! I absolutely love it.",
             "Terrible. DO NOT BUY THIS GARBAGE!!!!!!"]:
    votes=[fn(text) for fn in fns]
    pos_score=sum(w for v,w in zip(votes,weights) if v=="POS")
    neg_score=sum(w for v,w in zip(votes,weights) if v=="NEG")
    decision="POS" if pos_score>=neg_score else "NEG"
    print(f"  '{text[:40]}'  votes={votes}  decision={decision}")

print("\n=== Ex 5: Supervised Pipeline ===")
class Supervised:
    def __init__(self,agents):
        self.agents=agents
    def quality(self,r): return len(r)>20 and "error" not in r.lower()
    def run(self,inp):
        text=inp
        for agent in self.agents:
            for attempt in range(3):
                result=agent.respond(text)
                if self.quality(result): text=result; print(f"  [{agent.name}] attempt {attempt+1}: OK"); break
                print(f"  [{agent.name}] attempt {attempt+1}: FAILED quality check, retrying")
            else: print(f"  [{agent.name}] max retries, using last result")
        return text
bad_fn=lambda p: "err" if len(p)<30 else f"Processed: {p[:50]}"
agents_sup=[Agent("A","write",lambda p:f"Written content about: {p[:30]}"),
            Agent("B","check",bad_fn)]
Supervised(agents_sup).run("Explain quantum computing in simple terms")

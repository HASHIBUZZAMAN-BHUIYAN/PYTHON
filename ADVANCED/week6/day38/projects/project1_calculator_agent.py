"""
Project: Smart Calculator Agent
Teaches: building an agent that parses and solves multi-step math word problems
         using a tool registry with arithmetic, stats, and unit conversion tools.
~10 MB RAM, <1s on CPU
"""
import re, math, functools

class ToolError(Exception): pass

class ToolRegistry:
    def __init__(self): self._tools={}; self.log=[]
    def register(self,name,desc,param_types=None):
        def dec(fn):
            self._tools[name]={"fn":fn,"desc":desc,"ptypes":param_types or {}}
            return fn
        return dec
    def call(self,name,**kwargs):
        if name not in self._tools: raise ToolError(f"Unknown: {name}")
        t=self._tools[name]; cast={k:t["ptypes"].get(k,str)(v) for k,v in kwargs.items()}
        result=t["fn"](**cast); self.log.append({"tool":name,"kwargs":cast,"result":result}); return result

reg=ToolRegistry()

@reg.register("add","Add two numbers",{"a":float,"b":float})
def add(a,b): return a+b

@reg.register("subtract","Subtract b from a",{"a":float,"b":float})
def subtract(a,b): return a-b

@reg.register("multiply","Multiply two numbers",{"a":float,"b":float})
def multiply(a,b): return a*b

@reg.register("divide","Divide a by b",{"a":float,"b":float})
def divide(a,b):
    if abs(b)<1e-10: raise ToolError("Division by zero")
    return a/b

@reg.register("power","a raised to power b",{"a":float,"b":float})
def power(a,b): return a**b

@reg.register("sqrt","Square root of x",{"x":float})
def sqrt(x):
    if x<0: raise ToolError("Cannot take sqrt of negative")
    return math.sqrt(x)

@reg.register("percent","Compute p% of n",{"n":float,"p":float})
def percent(n,p): return n*p/100

@reg.register("mean","Mean of comma-separated numbers",{"nums":str})
def mean(nums): vals=[float(v) for v in nums.split(",")]; return sum(vals)/len(vals)

@reg.register("convert_km_miles","Convert km to miles",{"km":float})
def km_to_miles(km): return km*0.621371

@reg.register("convert_c_f","Convert Celsius to Fahrenheit",{"c":float})
def c_to_f(c): return c*9/5+32

# ─── Word problem solver ──────────────────────────────────────────────────────
PROBLEMS = [
    {
        "question": "A store sells 45 items at $12 each. What are the total sales? How much is a 15% discount?",
        "steps": [
            ("multiply",    {"a": 45, "b": 12}),
            ("percent",     {"n": 540, "p": 15}),
        ],
        "explain": ["Total sales = 45 × $12", "15% discount = 15% of $540"],
    },
    {
        "question": "A car travels 120 km. Convert to miles. If the trip takes 1.5 hours, what's the speed in mph?",
        "steps": [
            ("convert_km_miles", {"km": 120}),
            ("divide",           {"a": 74.5644, "b": 1.5}),
        ],
        "explain": ["120 km to miles", "speed = distance / time"],
    },
    {
        "question": "Temperature drops from 35°C to 8°C. Convert both to Fahrenheit. What is the difference?",
        "steps": [
            ("convert_c_f", {"c": 35}),
            ("convert_c_f", {"c": 8}),
            ("subtract",    {"a": 95.0, "b": 46.4}),
        ],
        "explain": ["35°C to °F", "8°C to °F", "difference in °F"],
    },
]

print("=== Smart Calculator Agent ===\n")
for prob in PROBLEMS:
    print(f"Q: {prob['question']}")
    for step_args, explain in zip(prob["steps"], prob["explain"]):
        tool_name, kwargs = step_args
        try:
            result = reg.call(tool_name, **{k:str(v) for k,v in kwargs.items()})
            print(f"  {explain} → {result:.4f}")
        except ToolError as e:
            print(f"  ERROR: {e}")
    print()

print("=== Tool Call Log ===")
for entry in reg.log:
    print(f"  {entry['tool']}({entry['kwargs']}) = {entry['result']:.4f}")

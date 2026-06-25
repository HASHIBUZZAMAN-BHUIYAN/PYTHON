# AGENTIC_AI Reference — Simple Agent Templates
# Three patterns: Reflex, FSM, Goal-based

import random

# ─── 1. REFLEX AGENT ─────────────────────────────────────────────────────────
class ReflexAgent:
    """
    Simple condition-action rules.
    Rules are evaluated in priority order (top wins).
    """
    def __init__(self):
        # (condition_fn, action_str) — evaluated in order
        self.rules = []

    def add_rule(self, condition_fn, action):
        self.rules.append((condition_fn, action))

    def step(self, percept):
        for condition, action in self.rules:
            if condition(percept):
                return action
        return "NO_OP"

# Usage:
# agent = ReflexAgent()
# agent.add_rule(lambda p: p["battery"] < 20, "CHARGE")
# agent.add_rule(lambda p: p["obstacle"], "TURN")
# agent.add_rule(lambda p: p["on_target"], "PICK")
# agent.add_rule(lambda p: True, "MOVE_FORWARD")


# ─── 2. FSM AGENT ─────────────────────────────────────────────────────────────
class FSMAgent:
    """
    Finite-state machine agent.
    Define states and transitions in subclass or via add_transition().
    """
    def __init__(self, initial_state):
        self.state = initial_state
        self._transitions = {}   # (state, condition_fn) -> (next_state, action_fn)

    def add_transition(self, from_state, condition_fn, to_state, action_fn=None):
        self._transitions.setdefault(from_state, []).append((condition_fn, to_state, action_fn))

    def step(self, percept):
        for cond, next_state, action_fn in self._transitions.get(self.state, []):
            if cond(percept):
                prev = self.state
                self.state = next_state
                result = action_fn(percept) if action_fn else f"{prev}→{next_state}"
                return result
        return f"stay:{self.state}"


# ─── 3. GOAL-BASED AGENT ─────────────────────────────────────────────────────
class GoalBasedAgent:
    """
    Agent with a goal stack.
    Pops goals and executes them, adding sub-goals as needed.
    Subclass and override execute_goal().
    """
    def __init__(self):
        self.goal_stack = []
        self.memory     = {}

    def push_goal(self, goal):  self.goal_stack.append(goal)
    def pop_goal(self):         return self.goal_stack.pop() if self.goal_stack else None

    def step(self, percept):
        goal = self.goal_stack[-1] if self.goal_stack else None
        if goal is None: return "IDLE"
        return self.execute_goal(goal, percept)

    def execute_goal(self, goal, percept):
        raise NotImplementedError("Override in subclass")


# ─── DEMO ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=== Reflex Agent Demo ===")
    agent = ReflexAgent()
    agent.add_rule(lambda p: p.get("battery",100)<20, "CHARGE")
    agent.add_rule(lambda p: p.get("obstacle"), "TURN")
    agent.add_rule(lambda p: p.get("on_target"), "PICK_UP")
    agent.add_rule(lambda p: True, "MOVE_FORWARD")

    for i in range(6):
        p = {"battery":max(0,90-i*15),"obstacle":i==2,"on_target":i==4}
        print(f"  Step {i}: {p} → {agent.step(p)}")

    print("\n=== FSM Agent Demo ===")
    fsm = FSMAgent("IDLE")
    fsm.add_transition("IDLE",    lambda p: p.get("order"),    "NAVIGATE", lambda p: f"going to {p['order']}")
    fsm.add_transition("NAVIGATE",lambda p: p.get("arrived"),  "PICK",     lambda p: "picking item")
    fsm.add_transition("PICK",    lambda p: True,              "RETURN",   lambda p: "returning")
    fsm.add_transition("RETURN",  lambda p: p.get("at_base"),  "IDLE",     lambda p: "depositing")

    events = [
        {"order":"A1","arrived":False,"at_base":False},
        {"order":"A1","arrived":True, "at_base":False},
        {"order":None,"arrived":False,"at_base":False},
        {"order":None,"arrived":False,"at_base":True},
    ]
    for e in events:
        print(f"  [{fsm.state}] {e} → {fsm.step(e)} [{fsm.state}]")

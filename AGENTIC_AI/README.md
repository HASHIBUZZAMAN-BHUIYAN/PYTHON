# AGENTIC_AI Reference Folder

Templates for building AI agents.

| File | Contents |
|------|----------|
| `simple_agent_template.py`              | Rule-based FSM agent, goal-based agent, vacuum agent |
| `llm_tool_calling_template.py`          | ReAct agent with tools, mock + real Anthropic API |
| `agent_memory_template.py`              | Conversation buffer, JSON fact store, TF-IDF memory search |
| `multi_agent_orchestration_template.py` | Pipeline, debate, and parallel multi-agent patterns |
| `tool_selection_template.py`            | Dynamic tool selection, chaining, retry, logging |

---

## Runnable Demo Agents

Full standalone mini-projects — run with no API key required (fully offline).

| File | What it does | Run |
|------|-------------|-----|
| `study_buddy_agent.py`        | Reads a passage, auto-generates quiz questions, grades answers, runs spaced-repetition flashcard scheduler | `python study_buddy_agent.py` |
| `text_reply_agent.py`         | Classifies incoming customer messages (complaint/question/refund/compliment/spam) and drafts appropriate replies; guardrail quarantines abuse | `python text_reply_agent.py` |
| `personal_assistant_agent.py` | In-memory to-do list with deadline+importance prioritization, daily focus plan, and weekly summary | `python personal_assistant_agent.py` |
| `code_review_agent.py`        | Static rule-based code review: flags mutable defaults, bare excepts, hardcoded secrets, None comparisons, debug prints, shadowed builtins; grades A–D | `python code_review_agent.py` |
| `negotiation_agent.py`        | Buyer and Seller agents negotiate over price using linear/exponential/cooperative concession strategies; prints full transcript and outcome | `python negotiation_agent.py` |

All demo agents run in `AGENTIC_AI\.venv`. No API key needed.

---

## Environment Setup

This folder has its **own dedicated virtual environment** (`AGENTIC_AI\.venv`) — separate from every other folder including BEGINNER and ADVANCED.

**From a fresh terminal:**
```
cd C:\Users\zen\Documents\GitHub\PYTHON
AGENTIC_AI\.venv\Scripts\activate
python AGENTIC_AI\simple_agent_template.py
```

**Or:** double-click `AGENTIC_AI\activate.bat` — it activates the venv and sets the working directory automatically.

Installed packages (see `AGENTIC_AI\requirements.txt`): numpy, scikit-learn

Note: `llm_tool_calling_template.py` optionally uses `anthropic` (real API calls). It runs fully offline in mock mode without any API key — no need to install `anthropic`.

---

## Related lessons
- ADVANCED/week3/day19 — Rule-based agents
- ADVANCED/week3/day20 — LLM-powered agents
- ADVANCED/week3/day21 — Capstone agentic system
- ADVANCED/week6/day36 — Agent memory
- ADVANCED/week6/day37 — Multi-step planning
- ADVANCED/week6/day38 — Tool-using agents II
- ADVANCED/week6/day39 — Multi-agent systems
- ADVANCED/week6/day40 — Evaluation & guardrails
- ADVANCED/week6/day42 — Final capstone

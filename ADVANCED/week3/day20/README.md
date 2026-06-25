# Advanced Day 20 — LLM-Powered Agents

## Objectives
- Understand ReAct (Reason + Act) agent architecture
- Build a tool-calling agent with mock/fallback mode (no API key required)
- Implement tool use: calculator, web-search mock, file-read mock
- Add conversation memory and multi-turn planning
- OPTIONAL: wire up real Anthropic API (set ANTHROPIC_API_KEY env var)

## How to run
```powershell
# No API key needed — runs in full mock mode
python ADVANCED\week3\day20\lesson.py
python ADVANCED\week3\day20\mini_project.py

# Optional: real API
$env:ANTHROPIC_API_KEY = "sk-ant-..."
python ADVANCED\week3\day20\lesson.py
```

## Time estimate
~75 minutes

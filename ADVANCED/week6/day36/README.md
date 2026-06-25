# Day 36 — Agent Memory

## Learning Objectives
- Understand short-term (conversation buffer) vs long-term (key-value JSON) memory
- Implement TF-IDF cosine similarity for semantic memory search
- Build memory consolidation: when buffer fills, compress to long-term store

## Topics Covered
1. **ConversationBuffer** — fixed-size rolling window of the last N turns
2. **Key-Value Store** — persistent JSON file for fact storage and recall
3. **Vector Memory** — TF-IDF + cosine similarity to find relevant past memories
4. **Memory Consolidation** — summarise overflow turns into a long-term entry

## Files
| File | Description |
|------|-------------|
| `lesson.py` | Full walkthrough of all four memory types |
| `exercises.py` | 5 practice TODOs |
| `solutions.py` | Complete solutions to all exercises |
| `projects/project1_memory_chatbot.py` | Chatbot with 5-turn rolling window |
| `projects/project2_fact_memory_agent.py` | Persistent JSON fact store agent |
| `projects/project3_memory_faq_agent.py` | TF-IDF FAQ search agent |

## Hardware
- CPU-only (Ryzen 7, 8 GB RAM, no GPU)
- No API key required — all projects use deterministic mock LLM

## Dependencies
```
pip install scikit-learn
```
(numpy ships with scikit-learn; no other heavy deps)

## Run
```
python lesson.py
python projects/project1_memory_chatbot.py
python projects/project2_fact_memory_agent.py
python projects/project3_memory_faq_agent.py
```

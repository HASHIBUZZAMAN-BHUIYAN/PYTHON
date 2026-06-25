# ~120 MB RAM, ~2s on CPU
"""
Project 2 — Fact Memory Agent
==============================
An agent with a persistent JSON fact-memory file.
Operations supported:
  STORE(key, value)  — save a fact
  RECALL(key)        — retrieve a fact by key
  LIST               — show all stored facts
  DELETE(key)        — remove a fact

Simulates 8 operations: store 5 facts, recall 3.
Reads and writes to memory.json in the same directory as this file.
No API key required.
"""

import json
import os
import time

# ──────────────────────────────────────────────
# PERSISTENT FACT STORE
# ──────────────────────────────────────────────

MEMORY_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "memory.json")


class FactMemoryAgent:
    """
    Agent with a persistent JSON-backed key-value fact store.
    All operations are logged to the console.
    """

    def __init__(self, filepath: str = MEMORY_FILE):
        self.filepath = filepath
        self._facts: dict = {}
        self._load()
        print(f"  [INIT] Fact memory loaded from: {self.filepath}")
        print(f"         {len(self._facts)} existing fact(s) found.\n")

    # ── Persistence ──────────────────────────────────────────

    def _load(self) -> None:
        if os.path.exists(self.filepath):
            with open(self.filepath, "r", encoding="utf-8") as f:
                self._facts = json.load(f)
        else:
            self._facts = {}

    def _save(self) -> None:
        with open(self.filepath, "w", encoding="utf-8") as f:
            json.dump(self._facts, f, indent=2)

    # ── Operations ───────────────────────────────────────────

    def store(self, key: str, value: str) -> None:
        ts = time.strftime("%H:%M:%S")
        self._facts[key] = {"value": value, "stored_at": ts}
        self._save()
        print(f"  [STORE  ] {key!r:30s} ← {value!r}")

    def recall(self, key: str) -> str | None:
        if key in self._facts:
            value = self._facts[key]["value"]
            stored_at = self._facts[key]["stored_at"]
            print(f"  [RECALL ] {key!r:30s} → {value!r}  (stored {stored_at})")
            return value
        else:
            print(f"  [RECALL ] {key!r:30s} → (NOT FOUND)")
            return None

    def list_all(self) -> None:
        print(f"  [LIST   ] {len(self._facts)} fact(s) in memory:")
        if not self._facts:
            print("            (empty)")
        for k, v in self._facts.items():
            print(f"            {k!r:30s} = {v['value']!r}")

    def delete(self, key: str) -> None:
        if key in self._facts:
            del self._facts[key]
            self._save()
            print(f"  [DELETE ] {key!r:30s} ← removed")
        else:
            print(f"  [DELETE ] {key!r:30s} ← not found, nothing removed")

    def clear_all(self) -> None:
        """Remove all facts and the JSON file (used for demo cleanup)."""
        self._facts = {}
        if os.path.exists(self.filepath):
            os.remove(self.filepath)
        print("  [CLEAR  ] All facts removed and memory.json deleted.")


# ──────────────────────────────────────────────
# SIMULATION — 8 operations
# ──────────────────────────────────────────────

def run_simulation():
    print("=" * 65)
    print("  FACT MEMORY AGENT — 8-operation simulation")
    print("=" * 65 + "\n")

    agent = FactMemoryAgent()

    # ── STORE 5 facts ──────────────────────────────────────

    print("  ── Phase 1: Store 5 facts ──────────────────────────────")
    agent.store("capital_of_france",   "Paris")
    agent.store("pi_approximation",    "3.14159265")
    agent.store("python_creator",      "Guido van Rossum")
    agent.store("speed_of_light_ms",   "299,792,458 m/s")
    agent.store("days_in_week",        "7")

    print()
    agent.list_all()

    # ── RECALL 3 facts ─────────────────────────────────────

    print("\n  ── Phase 2: Recall 3 facts ─────────────────────────────")
    agent.recall("capital_of_france")
    agent.recall("python_creator")
    agent.recall("unknown_key_xyz")     # intentional miss

    # ── Final summary ───────────────────────────────────────
    print("\n  ── Final state ─────────────────────────────────────────")
    agent.list_all()

    print("\n  ── Cleanup (removing memory.json for clean re-run) ─────")
    agent.clear_all()

    print("\n" + "=" * 65)
    print("  Simulation complete.")
    print("=" * 65)


if __name__ == "__main__":
    run_simulation()

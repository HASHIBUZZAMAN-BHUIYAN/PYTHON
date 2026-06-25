"""
Personal Productivity Assistant Agent
=======================================
What it does:
  Manages an in-memory task list and simulated reminders.
  "Today" is a fixed demo date (no real clock integration).
  The agent can:
    - add tasks (title, deadline, importance, category)
    - prioritize: ranks by deadline urgency * importance weight
    - answer "what should I do today?" with a sorted action plan
    - mark tasks complete
    - generate a weekly summary (completed vs pending)
    - set simple time-based reminders (stored, printed when "due")

  Demo run: pre-loads 8 sample tasks, shows the priority plan,
  marks 3 tasks complete, then re-summarizes the week.

What it teaches:
  - Stateful agent (in-memory task store across multiple calls)
  - Priority function (deadline proximity * importance -> urgency score)
  - Perceive -> plan -> act loop with task state transitions
  - Clean formatted CLI output without any UI library

How to run:
  python personal_assistant_agent.py

API key needed? NO -- fully offline. No API key required.
"""

from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Optional
import textwrap

# ─── DATA MODELS ──────────────────────────────────────────────────────────────

IMPORTANCE = {"critical": 4, "high": 3, "medium": 2, "low": 1}
CATEGORIES  = {"work", "personal", "health", "learning", "finance", "home"}

@dataclass
class Task:
    task_id:    str
    title:      str
    deadline:   date
    importance: str             # critical / high / medium / low
    category:   str
    notes:      str     = ""
    done:       bool    = False
    done_date:  Optional[date] = None

    def urgency(self, today: date) -> float:
        """Higher score = needs doing sooner. Overdue tasks score very high."""
        days_left = (self.deadline - today).days
        if days_left < 0:
            return 100.0 + IMPORTANCE[self.importance] * 10   # overdue
        if days_left == 0:
            return 50.0 + IMPORTANCE[self.importance] * 5     # due today
        # 1 / days_left scales urgency; multiply by importance weight
        return (1.0 / days_left) * IMPORTANCE[self.importance] * 10

@dataclass
class Reminder:
    reminder_id: str
    message:     str
    due_date:    date


# ─── PERSONAL ASSISTANT AGENT ─────────────────────────────────────────────────

class PersonalAssistantAgent:
    """
    State: tasks dict + reminders list + simulated "today" date
    Loop : perceive command -> decide action -> act on state -> report
    """

    def __init__(self, today: date):
        self.today:     date          = today
        self.tasks:     dict[str, Task]   = {}
        self.reminders: list[Reminder]    = []
        self._next_id:  int           = 1

    # ── Helpers ───────────────────────────────────────────────────────────────
    def _new_id(self, prefix: str = "T") -> str:
        tid = f"{prefix}{self._next_id:03d}"
        self._next_id += 1
        return tid

    def _deadline_label(self, task: Task) -> str:
        delta = (task.deadline - self.today).days
        if delta < 0:   return f"OVERDUE {abs(delta)}d ago"
        if delta == 0:  return "DUE TODAY"
        if delta == 1:  return "Tomorrow"
        return f"in {delta} days"

    # ── Perceive -> Add ───────────────────────────────────────────────────────
    def add_task(self, title: str, deadline: date, importance: str = "medium",
                 category: str = "work", notes: str = "") -> Task:
        """Add a task. Returns the created Task."""
        assert importance in IMPORTANCE, f"importance must be one of {list(IMPORTANCE)}"
        task = Task(
            task_id    = self._new_id("T"),
            title      = title,
            deadline   = deadline,
            importance = importance,
            category   = category,
            notes      = notes,
        )
        self.tasks[task.task_id] = task
        return task

    def add_reminder(self, message: str, due_date: date) -> Reminder:
        r = Reminder(self._new_id("R"), message, due_date)
        self.reminders.append(r)
        return r

    # ── Decide -> Prioritize ──────────────────────────────────────────────────
    def priority_plan(self, include_done: bool = False) -> list[Task]:
        """
        Return tasks sorted by urgency score (highest first).
        Urgency = f(deadline proximity, importance weight).
        """
        tasks = [t for t in self.tasks.values()
                 if (not t.done) or include_done]
        return sorted(tasks, key=lambda t: t.urgency(self.today), reverse=True)

    # ── Decide -> Today's plan ────────────────────────────────────────────────
    def plan_today(self) -> list[Task]:
        """Tasks due today or overdue that are not done."""
        return [t for t in self.priority_plan()
                if (t.deadline - self.today).days <= 0]

    # ── Act -> Complete ───────────────────────────────────────────────────────
    def complete(self, task_id: str):
        if task_id in self.tasks:
            self.tasks[task_id].done      = True
            self.tasks[task_id].done_date = self.today

    # ── Act -> Week summary ───────────────────────────────────────────────────
    def week_summary(self) -> dict:
        week_start = self.today - timedelta(days=self.today.weekday())
        week_end   = week_start + timedelta(days=6)
        completed  = [t for t in self.tasks.values()
                      if t.done and t.done_date and week_start <= t.done_date <= week_end]
        pending    = [t for t in self.tasks.values() if not t.done]
        overdue    = [t for t in pending if t.deadline < self.today]
        by_cat     = {}
        for t in self.tasks.values():
            by_cat.setdefault(t.category, {"done": 0, "pending": 0})
            if t.done: by_cat[t.category]["done"] += 1
            else:      by_cat[t.category]["pending"] += 1
        return {
            "week":      f"{week_start.strftime('%b %d')} - {week_end.strftime('%b %d')}",
            "completed": completed,
            "pending":   pending,
            "overdue":   overdue,
            "by_cat":    by_cat,
        }

    # ── Due reminders ─────────────────────────────────────────────────────────
    def check_reminders(self) -> list[Reminder]:
        return [r for r in self.reminders if r.due_date <= self.today]

    # ── Print helpers ─────────────────────────────────────────────────────────
    def print_plan(self, tasks: list[Task], title: str):
        print(f"\n  {title}")
        print("  " + "-"*62)
        if not tasks:
            print("  (nothing to show)")
            return
        print(f"  {'#':<4}  {'ID':<6}  {'Title':<32}  {'Deadline':<14}  {'Pri'}")
        print("  " + "-"*62)
        for i, t in enumerate(tasks, 1):
            pri = t.importance.upper()[:4]
            dl  = self._deadline_label(t)
            done = " [DONE]" if t.done else ""
            print(f"  {i:<4}  {t.task_id:<6}  {t.title[:30]:<32}  {dl:<14}  {pri}{done}")

    def print_week_summary(self):
        s = self.week_summary()
        print(f"\n  WEEKLY SUMMARY  ({s['week']})")
        print("  " + "-"*62)
        print(f"  Completed this week : {len(s['completed'])}")
        print(f"  Still pending       : {len(s['pending'])}")
        print(f"  Overdue             : {len(s['overdue'])}")
        if s['overdue']:
            for t in s['overdue']:
                print(f"    [!] {t.title[:50]}  (was due {t.deadline})")
        print(f"\n  By category:")
        for cat, counts in sorted(s['by_cat'].items()):
            bar = "=" * counts['done'] + "-" * counts['pending']
            print(f"    {cat:<12}: done={counts['done']}  pending={counts['pending']}  [{bar}]")


# ─── DEMO ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    TODAY = date(2026, 6, 26)   # fixed demo date
    agent = PersonalAssistantAgent(today=TODAY)

    print("=" * 65)
    print(f"  Personal Assistant Demo  (today = {TODAY})")
    print("=" * 65)

    # ── Step 1: Load tasks ────────────────────────────────────────────────────
    print("\n  [STEP 1] Loading 8 sample tasks...")

    agent.add_task("Submit quarterly report",   TODAY,                       "critical", "work",
                   "Must include Q2 financials and forecast.")
    agent.add_task("Doctor appointment",        TODAY + timedelta(days=1),   "high",    "health")
    agent.add_task("Review PR for feature X",   TODAY + timedelta(days=2),   "high",    "work")
    agent.add_task("Read Chapter 5 of AI book", TODAY + timedelta(days=3),   "medium",  "learning")
    agent.add_task("Pay electricity bill",      TODAY - timedelta(days=1),   "critical","finance",
                   "OVERDUE - avoid late fee!")
    agent.add_task("Grocery shopping",          TODAY + timedelta(days=4),   "medium",  "personal")
    agent.add_task("Write blog post draft",     TODAY + timedelta(days=7),   "low",     "learning")
    agent.add_task("Fix leaking kitchen tap",   TODAY + timedelta(days=10),  "medium",  "home")

    agent.add_reminder("Stand-up at 10am", TODAY)
    agent.add_reminder("Call dentist to reschedule", TODAY + timedelta(days=2))

    # ── Step 2: Show priority plan ────────────────────────────────────────────
    plan = agent.priority_plan()
    agent.print_plan(plan, "FULL PRIORITY PLAN (highest urgency first)")

    # ── Step 3: What to do today? ─────────────────────────────────────────────
    today_tasks = agent.plan_today()
    agent.print_plan(today_tasks, "TODAY'S FOCUS LIST (due today or overdue)")

    # ── Step 4: Reminders ─────────────────────────────────────────────────────
    due_reminders = agent.check_reminders()
    if due_reminders:
        print(f"\n  REMINDERS DUE TODAY:")
        for r in due_reminders:
            print(f"    [!] {r.message}")

    # ── Step 5: Mark 3 tasks complete ─────────────────────────────────────────
    print("\n  [STEP 2] Marking 3 tasks complete: T001, T005, T003")
    agent.complete("T001")
    agent.complete("T005")
    agent.complete("T003")

    # ── Step 6: Show updated plan ─────────────────────────────────────────────
    updated_plan = agent.priority_plan()
    agent.print_plan(updated_plan, "UPDATED PLAN (after 3 completions)")

    # ── Step 7: Week summary ──────────────────────────────────────────────────
    agent.print_week_summary()

    print("\n  Done. All agent operations ran fully offline.")

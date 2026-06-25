"""
Study Buddy Agent
=================
What it does:
  Reads a short passage of text on any topic, then autonomously:
    1. Summarizes the passage (TF-IDF sentence scoring — no API needed)
    2. Generates multiple-choice and short-answer quiz questions from it
    3. Grades a set of hardcoded sample answers and reports a score
    4. Runs a spaced-repetition flashcard loop: wrong answers are
       re-shown sooner (simulated over loop "days", no real waiting)

What it teaches:
  - Perceive → Decide → Act agent loop
  - Simple NLP without external models (TF-IDF, regex)
  - Spaced-repetition scheduling logic (SM-2 simplified)
  - State tracking across multiple agent turns

How to run:
  python study_buddy_agent.py

API key needed? NO — runs fully offline. No API key required.
"""

import re
import math
from collections import defaultdict, Counter

# ─── NLP HELPERS (no external libraries needed) ───────────────────────────────

def tokenize(text: str) -> list[str]:
    return re.findall(r"\b[a-z]{3,}\b", text.lower())

def tfidf_scores(sentences: list[str]) -> list[float]:
    """Score each sentence by TF-IDF word importance (extractive summarization)."""
    all_words  = [tokenize(s) for s in sentences]
    doc_count  = len(sentences)
    df         = Counter(w for words in all_words for w in set(words))
    scores     = []
    for words in all_words:
        tf   = Counter(words)
        score = sum(tf[w] * math.log(doc_count / (df[w] + 1)) for w in tf)
        scores.append(score)
    total = sum(scores) or 1
    return [s / total for s in scores]

def summarize(text: str, n_sentences: int = 2) -> str:
    """Return the n top-scoring sentences as an extractive summary."""
    sents = [s.strip() for s in re.split(r"[.!?]+", text) if len(s.strip()) > 20]
    if len(sents) <= n_sentences:
        return " ".join(sents)
    scores  = tfidf_scores(sents)
    ranked  = sorted(zip(scores, range(len(sents))), reverse=True)
    top_idx = sorted(i for _, i in ranked[:n_sentences])
    return " ".join(sents[i] + "." for i in top_idx)

# ─── QUESTION BANK (auto-generated from passage) ──────────────────────────────

def extract_key_terms(text: str, n: int = 8) -> list[str]:
    """Pull the most frequent content words — become quiz anchors."""
    stop = {"the","and","for","are","was","were","that","this","with","have",
            "from","they","been","its","also","but","not","can","has","all",
            "into","which","through","during","these","those","such","more",
            "most","some","each","both","than","then","when","where","how",
            "what","who","why","two","one","used","uses","use","using","their"}
    words = [w for w in tokenize(text) if w not in stop]
    return [w for w, _ in Counter(words).most_common(n)]

def make_questions(passage: str, key_terms: list[str]) -> list[dict]:
    """
    Build a small question bank from the passage.
    Strategy: pick sentences containing key terms, blank out the term
    → fill-in-the-blank becomes a short-answer question.
    Also include 2 hardcoded multiple-choice about the topic.
    """
    sents   = [s.strip() for s in re.split(r"[.!?]+", passage) if len(s.strip()) > 30]
    questions = []

    # Fill-in-the-blank (short answer) questions
    for term in key_terms[:4]:
        for s in sents:
            if term in s.lower():
                blanked = re.sub(rf"\b{term}\b", "______", s, flags=re.IGNORECASE, count=1)
                questions.append({
                    "type":    "short_answer",
                    "q":       f"Fill in the blank: {blanked}",
                    "answer":  term,
                    "hint":    f"({len(term)} letters)",
                    "topic":   term,
                })
                break  # one question per term

    return questions

# ─── SPACED REPETITION SCHEDULER ──────────────────────────────────────────────

class FlashCard:
    """Minimal SM-2-inspired card: tracks interval and ease factor."""
    def __init__(self, question: str, answer: str):
        self.question   = question
        self.answer     = answer
        self.interval   = 1    # "days" until next review
        self.ease       = 2.5  # ease factor (higher = shown less often)
        self.due_day    = 0    # next review day (0 = show immediately)
        self.reviews    = 0

    def review(self, correct: bool):
        """Update schedule. Correct → longer interval. Wrong → reset."""
        self.reviews += 1
        if correct:
            if self.reviews == 1:
                self.interval = 1
            elif self.reviews == 2:
                self.interval = 3
            else:
                self.interval = round(self.interval * self.ease)
            self.ease = max(1.3, self.ease + 0.1)
        else:
            self.interval = 1           # wrong → see it again tomorrow
            self.ease     = max(1.3, self.ease - 0.2)

    def schedule(self, today: int):
        self.due_day = today + self.interval


class SpacedRepetitionScheduler:
    """Simulates 'days' as loop iterations. Picks due cards each day."""
    def __init__(self):
        self.cards: list[FlashCard] = []
        self.day = 0

    def add(self, question: str, answer: str):
        self.cards.append(FlashCard(question, answer))

    def due_today(self) -> list[FlashCard]:
        return [c for c in self.cards if c.due_day <= self.day]

    def advance_day(self):
        self.day += 1


# ─── GRADER ───────────────────────────────────────────────────────────────────

def grade_answer(user_answer: str, correct_answer: str) -> tuple[bool, str]:
    """Flexible grader: accept if answer contains correct term (case-insensitive)."""
    ua = user_answer.strip().lower()
    ca = correct_answer.strip().lower()
    correct = ca in ua or ua in ca or ua == ca
    feedback = "Correct!" if correct else f"Wrong. The answer was: {correct_answer}"
    return correct, feedback


# ─── STUDY BUDDY AGENT ────────────────────────────────────────────────────────

class StudyBuddyAgent:
    """
    Perceive–Decide–Act loop:
      Perceive : read passage + student answers
      Decide   : summarize / generate questions / grade / schedule
      Act      : print summary, quiz, score, flashcard plan
    """

    def __init__(self, topic: str, passage: str):
        self.topic      = topic
        self.passage    = passage
        self.score      = 0
        self.total      = 0
        self.scheduler  = SpacedRepetitionScheduler()

    # ── Perceive ──────────────────────────────────────────────────────────────
    def perceive(self) -> dict:
        key_terms = extract_key_terms(self.passage)
        questions = make_questions(self.passage, key_terms)
        summary   = summarize(self.passage, n_sentences=2)
        return {"summary": summary, "questions": questions, "key_terms": key_terms}

    # ── Decide ────────────────────────────────────────────────────────────────
    def decide_quiz(self, questions: list[dict], sample_answers: list[str]) -> list[dict]:
        """Grade each answer and prepare the result record."""
        results = []
        for q, ua in zip(questions, sample_answers):
            correct, feedback = grade_answer(ua, q["answer"])
            self.score += correct
            self.total += 1
            results.append({"q": q["q"], "student": ua, "correct": correct,
                             "feedback": feedback, "topic": q["topic"]})
        return results

    # ── Act ───────────────────────────────────────────────────────────────────
    def act_summary(self, summary: str):
        print("\n" + "-"*60)
        print(f"  TOPIC: {self.topic}")
        print("-"*60)
        print(f"  SUMMARY:\n  {summary}")

    def act_quiz(self, results: list[dict]):
        print(f"\n  QUIZ RESULTS  ({self.score}/{self.total} correct)")
        print("  " + "-"*56)
        for r in results:
            sym = "[OK]" if r["correct"] else "[X] "
            print(f"  {sym}  Q: {r['q'][:60]}")
            print(f"     Your answer: {r['student']}  -> {r['feedback']}")

    def act_flashcard_plan(self, results: list[dict]):
        """Load wrong answers into the spaced-repetition scheduler and simulate 3 days."""
        wrong = [r for r in results if not r["correct"]]
        if not wrong:
            print("\n  FLASHCARDS: All correct -- nothing to review!")
            return

        print(f"\n  FLASHCARD SCHEDULER  ({len(wrong)} cards added for wrong answers)")
        print("  " + "-"*56)
        for r in wrong:
            self.scheduler.add(r["q"], r["topic"])

        # Simulate 4 "days" of review
        for day in range(4):
            self.scheduler.day = day
            due = self.scheduler.due_today()
            if not due:
                print(f"  Day {day}: nothing due.")
                continue
            print(f"  Day {day}: {len(due)} card(s) due ->", end=" ")
            for card in due:
                # Simulate: student answers correctly on day 2+, wrong on day 0
                simulated_correct = day >= 2
                card.review(simulated_correct)
                card.schedule(day)
                sym = "[OK]" if simulated_correct else "[X] "
                print(f"{sym} [{card.question[:28]}... next in {card.interval}d]", end="  ")
            print()

    def run(self, sample_answers: list[str]):
        """Full agent loop: perceive -> decide -> act."""
        state = self.perceive()
        self.act_summary(state["summary"])
        print(f"\n  QUESTIONS ({len(state['questions'])} generated):")
        for i, q in enumerate(state["questions"], 1):
            print(f"  {i}. {q['q']}")
        results = self.decide_quiz(state["questions"], sample_answers)
        self.act_quiz(results)
        self.act_flashcard_plan(results)
        print(f"\n  FINAL SCORE: {self.score}/{self.total} "
              f"({self.score/max(self.total,1):.0%})")


# ─── DEMO ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    PASSAGE = """
    Photosynthesis is the process by which green plants convert sunlight into
    chemical energy stored as glucose. Chlorophyll, the green pigment found in
    chloroplasts, absorbs light energy primarily from the red and blue parts of
    the visible spectrum. During photosynthesis, plants take in carbon dioxide
    from the air and water from the soil. These raw materials are transformed
    into glucose and oxygen through two main stages: the light-dependent
    reactions, which produce ATP and NADPH, and the Calvin cycle, which uses
    that energy to fix carbon dioxide into glucose molecules. Oxygen is
    released as a by-product and enters the atmosphere. Photosynthesis is
    fundamental to life on Earth because it forms the base of almost all food
    chains and is the primary source of atmospheric oxygen.
    """

    # Hardcoded demo answers (simulate a student responding).
    # The agent extracts key terms: photosynthesis, energy, glucose, oxygen.
    # Answers below are deliberately mixed: some correct, some wrong,
    # so the flashcard scheduler has something to work with.
    SAMPLE_ANSWERS = [
        "photosynthesis",  # Q1 correct  (photosynthesis)
        "glucose",         # Q2 wrong    (correct is: energy)
        "glucose",         # Q3 correct  (glucose)
        "oxygen",          # Q4 correct  (oxygen)
    ]

    agent = StudyBuddyAgent(topic="Photosynthesis (Biology)", passage=PASSAGE)
    agent.run(SAMPLE_ANSWERS)

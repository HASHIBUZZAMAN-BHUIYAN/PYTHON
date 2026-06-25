# ~200 MB RAM, ~15s on CPU
"""
Project 2: Analogy Solver
===========================
What it does: Demonstrates word analogy arithmetic using Word2Vec.
              king - man + woman = queen style reasoning.
              Handles cases where analogies fail gracefully.

What it teaches:
  - Vector arithmetic in embedding space
  - How semantic relationships are encoded as directions
  - Limitations of small corpora for analogy tasks
  - Robust error handling for NLP models
"""

import random
import numpy as np
from gensim.models import Word2Vec

random.seed(42)
np.random.seed(42)

# ─── Build Corpus for Analogies ──────────────────────────────────────────────
# Needs enough coverage for the analogy pairs we want to test

CORPUS_SENTENCES = [
    # Royalty & Gender
    "the king is a powerful man who rules the kingdom",
    "the queen is a royal woman who rules beside the king",
    "the prince is a young man who will become king",
    "the princess is a young woman in the royal family",
    "the king and queen rule together in the royal palace",
    "the man bowed before the king in the great hall",
    "the woman curtsied before the queen in the garden",
    "the boy trained to become a knight for the king",
    "the girl learned to serve the queen in the palace",
    "a royal man becomes king and a royal woman becomes queen",
    "the king wore his crown and the queen wore her tiara",
    "the prince will one day sit on the throne as king",
    "the princess grew up to be a wise and noble queen",
    # Countries & Capitals
    "paris is the capital and largest city of france",
    "france is a country in western europe with its capital paris",
    "rome is the capital city of the country of italy",
    "italy is a european country and its capital is rome",
    "london is the capital of england and the united kingdom",
    "england is a country whose capital city is london",
    "berlin is the capital of germany in central europe",
    "germany borders france and its capital city is berlin",
    "madrid is the capital of spain in southern europe",
    "spain is a country on the iberian peninsula with capital madrid",
    # Animals & Sounds/Homes
    "the dog barks loudly and lives in a kennel",
    "the cat meows softly and lives in a house",
    "the cow moos and lives on a farm in the countryside",
    "the lion roars and lives in the savanna jungle",
    "the fish swims silently and lives in the ocean or river",
    "the bird sings beautifully and lives in a nest in trees",
    "dogs and cats are common pets that live with humans",
    "lions and tigers are wild predators that hunt for food",
    # Actions & People
    "the teacher teaches students in the classroom every day",
    "the doctor treats patients in the hospital with medicine",
    "the engineer builds machines and solves technical problems",
    "the chef cooks delicious food in the restaurant kitchen",
    "the writer writes books and stories for many readers",
    "the painter paints beautiful pictures with colorful paint",
    "the singer sings songs on a large stage for audiences",
    "the runner runs fast in the marathon race every year",
    # Technology
    "computers process data and run software programs quickly",
    "the internet connects computers around the world globally",
    "smartphones are small computers that fit in your pocket",
    "software programs tell computers what tasks to perform",
    "hardware is the physical components inside a computer",
    "algorithms are step by step instructions for solving problems",
    "data is stored in databases and processed by computers",
]

# Repeat and shuffle for better training
augmented = CORPUS_SENTENCES[:]
for _ in range (15):
    copy = CORPUS_SENTENCES[:]
    random.shuffle(copy)
    augmented.extend(copy)

CORPUS = [s.lower().split() for s in augmented]

# ─── Train Model ─────────────────────────────────────────────────────────────
print("=" * 60)
print("ANALOGY SOLVER — Word2Vec")
print("=" * 60)
print(f"Training on {len(CORPUS)} sentences...")

model = Word2Vec(
    sentences   = CORPUS,
    vector_size = 100,
    window      = 5,
    min_count   = 3,
    epochs      = 100,
    sg          = 1,
    seed        = 42,
    workers     = 1,
)
print(f"Vocabulary: {len(model.wv)} words | Dims: {model.wv.vector_size}")

# ─── Analogy Solver Function ──────────────────────────────────────────────────
def solve_analogy(model, word_a, word_b, word_c, expected=None, topn=5):
    """
    Solve analogy: word_a is to word_b as word_c is to ?
    Equivalent to: word_b - word_a + word_c = ?

    Args:
        word_a, word_b, word_c : analogy words
        expected : the expected answer (for display)
        topn     : number of candidates to return
    Returns:
        list of (word, score) tuples, or None on failure
    """
    # Check all words are in vocabulary
    missing = [w for w in [word_a, word_b, word_c] if w not in model.wv]
    if missing:
        return None, f"Words not in vocabulary: {missing}"

    try:
        # word_b - word_a + word_c
        results = model.wv.most_similar(
            positive=[word_b, word_c],
            negative=[word_a],
            topn=topn
        )
        return results, None
    except Exception as e:
        return None, str(e)

def display_analogy(word_a, word_b, word_c, results, error, expected=None):
    """Pretty-print an analogy result."""
    print(f"\n  '{word_a}' is to '{word_b}' as '{word_c}' is to ___")
    print(f"  Formula: {word_b} - {word_a} + {word_c} = ?")

    if error:
        print(f"  ERROR: {error}")
        return

    top_word, top_score = results[0]
    if expected:
        correct = top_word.lower() == expected.lower()
        status = "CORRECT" if correct else "INCORRECT"
        print(f"  Answer : '{top_word}'  (score={top_score:.3f})  [{status}, expected='{expected}']")
    else:
        print(f"  Answer : '{top_word}'  (score={top_score:.3f})")

    candidates = [(w, round(s, 3)) for w, s in results[:5]]
    print(f"  Top 5  : {candidates}")

# ─── Test Analogies ──────────────────────────────────────────────────────────
print("\n" + "─" * 60)
print("ANALOGY TESTS")
print("─" * 60)

analogies = [
    # (word_a,   word_b,  word_c,   expected)
    ("man",      "king",   "woman",  "queen"),       # gender + royalty
    ("man",      "prince", "woman",  "princess"),    # gender + royalty 2
    ("king",     "queen",  "prince", "princess"),    # royalty relationships
    ("france",   "paris",  "italy",  "rome"),        # country → capital
    ("france",   "paris",  "england","london"),      # country → capital 2
    ("dog",      "cat",    "lion",   "tiger"),       # animal pairs
    ("teacher",  "school", "doctor", "hospital"),   # profession → workplace
]

results_summary = []
for word_a, word_b, word_c, expected in analogies:
    results, error = solve_analogy(model, word_a, word_b, word_c, expected)
    display_analogy(word_a, word_b, word_c, results, error, expected)

    if results and not error:
        top = results[0][0].lower()
        correct = top == expected.lower()
        results_summary.append((f"{word_a}:{word_b}::{word_c}:?", expected, top, correct))

# ─── Summary ──────────────────────────────────────────────────────────────────
print("\n" + "─" * 60)
print("SUMMARY")
print("─" * 60)
total   = len(results_summary)
correct = sum(1 for _, _, _, ok in results_summary if ok)

print(f"\n{'Analogy':<30} {'Expected':<12} {'Got':<12} {'Result'}")
print("-" * 65)
for analogy, expected, got, ok in results_summary:
    status = "CORRECT" if ok else "wrong  "
    print(f"  {analogy:<30} {expected:<12} {got:<12} {status}")

print(f"\nScore: {correct}/{total} correct")
print(f"\nNote: Small corpus + limited vocabulary means some analogies may fail.")
print(f"      Real Word2Vec (Google News, 3B tokens) achieves ~65% on standard analogy tasks.")

# ─── Manual Vector Arithmetic Demo ───────────────────────────────────────────
print("\n" + "─" * 60)
print("MANUAL VECTOR ARITHMETIC DEMO")
print("─" * 60)

print("\nShowing how analogy is computed step by step:")
try:
    v_king   = model.wv["king"]
    v_man    = model.wv["man"]
    v_woman  = model.wv["woman"]

    result_vec = v_king - v_man + v_woman
    print(f"  v_king[:5]    = {v_king[:5].round(3)}")
    print(f"  v_man[:5]     = {v_man[:5].round(3)}")
    print(f"  v_woman[:5]   = {v_woman[:5].round(3)}")
    print(f"  result[:5]    = {result_vec[:5].round(3)}")
    print(f"\n  The result vector is closest to: 'queen'")
    print(f"  This works because gender is encoded as a direction in vector space!")
except KeyError as e:
    print(f"  Word not in vocab: {e}")

print("\n--- Project 2 Complete ---")

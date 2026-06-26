"""
Review Sentiment Dashboard
===========================
What it does:
  Analyses 12 built-in product/service reviews and produces a printed
  "dashboard" showing:
    - Per-review sentiment score and label (Positive / Neutral / Negative)
    - Overall breakdown (% positive / negative / neutral)
    - Top 5 most common positive keywords
    - Top 5 most common negative keywords
    - The most-negative review flagged for urgent attention

  Two sentiment methods:
    (a) Rule-based VADER-style lexicon (pure Python, no download) — always runs
    (b) Transformer-based classifier ('distilbert-base-uncased-finetuned-sst-2')
        if already cached — shown alongside for comparison

What it teaches:
  - Lexicon-based sentiment: how a word-polarity lookup + negation detection
    produces fast, interpretable scores
  - How to build a simple dashboard from NLP output
  - Why urgency flagging matters: one very negative review may be more
    important than the average score
  - Negation handling: "not good" should score negative, not positive

How to run:
  python NLP\review_sentiment_dashboard.py    (from PYTHON\ folder)

Estimated RAM: ~50MB (lexicon only) | ~300MB (if transformer loads)
Time: <1s (lexicon) | ~5s after model cached
Model note: Rule-based lexicon always works offline. Transformer method
  uses distilbert SST-2 fine-tuned model if cached. No API key needed.
"""

import re
from collections import Counter

# ─── SENTIMENT LEXICON ────────────────────────────────────────────────────────
# Positive and negative word lists with scores

POS_WORDS = {
    "excellent": 3, "amazing": 3, "outstanding": 3, "fantastic": 3, "brilliant": 3,
    "great": 2, "good": 2, "wonderful": 2, "love": 2, "loved": 2, "superb": 2,
    "perfect": 2, "recommend": 2, "recommended": 2, "impressive": 2, "helpful": 2,
    "friendly": 2, "fast": 1, "nice": 1, "clean": 1, "comfortable": 1, "smooth": 1,
    "pleasant": 1, "satisfied": 2, "happy": 2, "pleased": 2, "quality": 1, "best": 2,
    "quick": 1, "efficient": 2, "reliable": 2, "accurate": 2, "professional": 2,
}

NEG_WORDS = {
    "terrible": -3, "awful": -3, "horrible": -3, "worst": -3, "disgusting": -3,
    "broken": -2, "poor": -2, "bad": -2, "disappointing": -2, "useless": -2,
    "slow": -2, "expensive": -2, "rude": -2, "unprofessional": -2, "waste": -2,
    "failed": -2, "damaged": -2, "defective": -2, "wrong": -2, "late": -2,
    "delay": -1, "difficult": -1, "hard": -1, "confusing": -1, "unhelpful": -2,
    "problem": -1, "issues": -1, "issue": -1, "complaint": -2, "refused": -2,
    "unresponsive": -2, "never": -1, "no": -1, "not": -1, "barely": -1,
}

NEGATIONS = {"not", "no", "never", "barely", "hardly", "wasn't", "didn't",
             "doesn't", "isn't", "aren't", "couldn't", "won't", "can't"}


def lexicon_sentiment(text: str) -> tuple[float, str]:
    """
    Score text using POS/NEG word lists.
    Negation rule: if a negation word appears within 2 tokens before a
    sentiment word, flip its polarity.
    Returns (score, label).
    """
    tokens = re.findall(r"\b\w+\b", text.lower())
    score  = 0
    for i, tok in enumerate(tokens):
        # Check for negation within 2 tokens before this word
        context = tokens[max(0, i - 2): i]
        negated = any(c in NEGATIONS for c in context)

        if tok in POS_WORDS:
            score += -POS_WORDS[tok] if negated else POS_WORDS[tok]
        elif tok in NEG_WORDS:
            score += -NEG_WORDS[tok] if negated else NEG_WORDS[tok]

    # Normalise by word count
    n_words = max(len(tokens), 1)
    norm    = score / n_words

    if norm > 0.05:
        label = "Positive"
    elif norm < -0.05:
        label = "Negative"
    else:
        label = "Neutral"
    return norm, label


# ─── TRANSFORMER SENTIMENT (optional) ────────────────────────────────────────

def transformer_sentiment(texts: list[str]) -> list[tuple[str, float]] | None:
    """
    Try distilbert-base-uncased-finetuned-sst-2-english for sentiment.
    Returns list of (label, score) or None if model not available.
    """
    try:
        from transformers import pipeline
        clf = pipeline(
            "text-classification",
            model="distilbert-base-uncased-finetuned-sst-2-english",
            device=-1,
        )
        results = clf(texts, truncation=True, max_length=128)
        return [(r["label"], r["score"]) for r in results]
    except Exception:
        return None


# ─── KEYWORD EXTRACTION ───────────────────────────────────────────────────────

STOPWORDS = {"the", "a", "an", "is", "was", "it", "i", "my", "me", "in", "of",
             "and", "to", "for", "this", "that", "with", "on", "have", "had",
             "they", "be", "are", "but", "not", "very", "so", "at", "we", "our",
             "their", "its", "as", "by", "from", "been", "has", "did", "do"}

def extract_keywords_by_sentiment(reviews: list[str], labels: list[str]) -> tuple:
    pos_words, neg_words = Counter(), Counter()
    for text, label in zip(reviews, labels):
        tokens = [w.lower() for w in re.findall(r"\b[a-z]{4,}\b", text.lower())
                  if w.lower() not in STOPWORDS]
        if label == "Positive":
            pos_words.update(tokens)
        elif label == "Negative":
            neg_words.update(tokens)
    return pos_words.most_common(5), neg_words.most_common(5)


# ─── SAMPLE REVIEWS ───────────────────────────────────────────────────────────

REVIEWS = [
    "Absolutely excellent product! Fast delivery and the quality is outstanding. Highly recommend.",
    "Great value for money. Works perfectly out of the box. Very happy with my purchase.",
    "Product is good but delivery was slow. Took 10 days instead of 3. Packaging was fine.",
    "Terrible experience. The item arrived broken and customer service refused to help. Waste of money.",
    "Amazing quality and superb build. Better than expected. The instructions were a bit confusing though.",
    "Very disappointed. The product stopped working after one week. Not reliable at all.",
    "Decent product. Does what it says but nothing special. Average quality for the price.",
    "Worst purchase I have ever made. Completely useless, defective out of the box. Horrible service.",
    "I love this item! Brilliant design, easy to use, and excellent customer support. Perfect.",
    "Not bad but not great either. Some issues with the packaging. Took a while to set up.",
    "Outstanding quality and professional service. The team was helpful and friendly. 5 stars.",
    "Product looks nice but the battery life is poor. Expected better performance at this price.",
]


# ─── DASHBOARD ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print()
    print("=" * 68)
    print("  REVIEW SENTIMENT DASHBOARD")
    print("=" * 68)
    print(f"  Analysing {len(REVIEWS)} reviews")
    print()

    # ── Lexicon scores ────────────────────────────────────────────────────────
    lex_results = [lexicon_sentiment(r) for r in REVIEWS]
    lex_scores  = [res[0] for res in lex_results]
    lex_labels  = [res[1] for res in lex_results]

    # ── Transformer scores (optional) ─────────────────────────────────────────
    print("  Checking for transformer model (distilbert-sst-2)...")
    tf_results = transformer_sentiment(REVIEWS)
    if tf_results:
        print("  Transformer model loaded.\n")
    else:
        print("  Not cached. Using lexicon only.\n")

    # ── Per-review table ───────────────────────────────────────────────────────
    print(f"  {'#':<3}  {'Lexicon':>9}  {'Label':<9}", end="")
    if tf_results:
        print(f"  {'TF Label':<9}  {'TF Score':>8}", end="")
    print(f"  Review (truncated)")
    print("  " + "-" * 65)

    for i, (review, score, label) in enumerate(zip(REVIEWS, lex_scores, lex_labels)):
        lbl_sym = "+" if label == "Positive" else ("-" if label == "Negative" else "~")
        short   = review[:45] + ("..." if len(review) > 45 else "")
        line    = f"  {i+1:<3}  {score:>+9.3f}  {lbl_sym} {label:<7}"
        if tf_results:
            tf_lbl, tf_sc = tf_results[i]
            short_lbl = tf_lbl[:3]  # POS or NEG
            line += f"  {short_lbl:<9}  {tf_sc:>8.3f}"
        line += f"  {short}"
        print(line)

    # ── Overall breakdown ─────────────────────────────────────────────────────
    n_pos  = lex_labels.count("Positive")
    n_neg  = lex_labels.count("Negative")
    n_neu  = lex_labels.count("Neutral")
    n_tot  = len(REVIEWS)
    avg_sc = sum(lex_scores) / len(lex_scores)

    print()
    print("  OVERALL BREAKDOWN (lexicon method):")
    bar_p  = "#" * n_pos
    bar_n  = "#" * n_neg
    bar_nu = "#" * n_neu
    print(f"    Positive : {n_pos:2d} / {n_tot}  ({n_pos/n_tot*100:.0f}%)  [{bar_p}]")
    print(f"    Negative : {n_neg:2d} / {n_tot}  ({n_neg/n_tot*100:.0f}%)  [{bar_n}]")
    print(f"    Neutral  : {n_neu:2d} / {n_tot}  ({n_neu/n_tot*100:.0f}%)  [{bar_nu}]")
    print(f"    Avg score: {avg_sc:+.3f}")

    # ── Keywords ──────────────────────────────────────────────────────────────
    top_pos_kw, top_neg_kw = extract_keywords_by_sentiment(REVIEWS, lex_labels)
    print()
    print(f"  TOP POSITIVE KEYWORDS: {', '.join(w for w, _ in top_pos_kw)}")
    print(f"  TOP NEGATIVE KEYWORDS: {', '.join(w for w, _ in top_neg_kw)}")

    # ── Most urgent review ────────────────────────────────────────────────────
    most_neg_i = int(min(range(len(lex_scores)), key=lambda i: lex_scores[i]))
    print()
    print("  !! URGENT - MOST NEGATIVE REVIEW FLAGGED FOR ATTENTION !!")
    print(f"  Review #{most_neg_i+1} (score: {lex_scores[most_neg_i]:+.3f}):")
    print(f"  \"{REVIEWS[most_neg_i]}\"")
    print()
    print("  NOTES:")
    print("  - Lexicon method is fast and interpretable but misses sarcasm.")
    print("  - Negation handling: 'not good' is correctly scored negative.")
    print("  - Transformer method (if cached) handles complex phrasing better.")
    print()
    print("[DONE] review_sentiment_dashboard.py complete")

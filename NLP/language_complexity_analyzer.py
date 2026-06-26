"""
Language Complexity Analyzer
==============================
What it does:
  Analyses text samples for readability and complexity using formulas
  implemented directly in Python (no heavy library needed). Metrics:
    - Average sentence length (words per sentence)
    - Average word length (characters per word)
    - Average syllables per word (approximated by vowel-group counting)
    - Flesch Reading Ease score (206.835 - 1.015*ASL - 84.6*ASW)
    - Flesch-Kincaid Grade Level (0.39*ASL + 11.8*ASW - 15.59)
    - Percentage of complex words (3+ syllables)
    - Type-Token Ratio (unique words / total words) — lexical diversity

  Compares 4 built-in text samples with varying complexity levels:
    - Children's story excerpt (simple)
    - News article excerpt (moderate)
    - Academic paper abstract (complex)
    - Legal contract excerpt (very complex)

What it teaches:
  - How readability formulas work: just arithmetic on sentence/word/syllable
    statistics — no ML needed
  - Why these metrics matter: writers, educators, accessibility tools
  - Syllable counting heuristic: count vowel groups (not perfect but good
    enough for English readability estimation)
  - Trade-offs: long sentences OR long words both reduce reading ease score

How to run:
  python NLP\language_complexity_analyzer.py    (from PYTHON\ folder)

Estimated RAM: <20MB | Time: <1s
Model note: 100% rule-based arithmetic. No model, no download, no API key.
"""

import re
import math
from collections import Counter


# ─── SYLLABLE COUNTING ────────────────────────────────────────────────────────

VOWEL_PAT = re.compile(r"[aeiou]+", re.IGNORECASE)


def count_syllables(word: str) -> int:
    """
    Approximate syllable count via vowel-group rule.
    Rule: count continuous vowel runs (aeiou), subtract silent-e, minimum 1.
    Not perfect but matches Flesch-Kincaid assumptions closely enough.
    """
    word  = word.lower().strip(".,!?;:'\"")
    if not word:
        return 0
    count = len(VOWEL_PAT.findall(word))
    # Silent-e at end of word (e.g. 'take', 'name') -> subtract 1
    if word.endswith("e") and count > 1:
        count -= 1
    # Every vowel run that ends in 'ed' is usually silent (e.g. 'changed')
    if word.endswith("ed") and count > 1:
        count -= 1
    return max(count, 1)


def count_syllables_text(text: str) -> list[int]:
    words = re.findall(r"\b[a-zA-Z]+\b", text)
    return [count_syllables(w) for w in words]


# ─── TEXT METRICS ─────────────────────────────────────────────────────────────

def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()]


def analyse_text(text: str) -> dict:
    sentences = split_sentences(text)
    words     = re.findall(r"\b[a-zA-Z]+\b", text)
    syllables = [count_syllables(w) for w in words]

    n_sents   = max(len(sentences), 1)
    n_words   = max(len(words), 1)

    asl  = n_words / n_sents        # Average Sentence Length
    asw  = sum(syllables) / n_words # Average Syllables per Word

    # Flesch Reading Ease (higher = easier; 0-100 scale)
    fre  = 206.835 - (1.015 * asl) - (84.6 * asw)
    fre  = max(0, min(121, fre))    # clamp to reasonable range

    # Flesch-Kincaid Grade Level (US school grade)
    fkgl = (0.39 * asl) + (11.8 * asw) - 15.59
    fkgl = max(0, fkgl)

    # Average word length (characters)
    avg_word_len = sum(len(w) for w in words) / n_words

    # Complex words: 3+ syllables
    complex_words = sum(1 for s in syllables if s >= 3)
    pct_complex   = complex_words / n_words * 100

    # Type-Token Ratio (lexical diversity)
    unique = len(set(w.lower() for w in words))
    ttr    = unique / n_words

    return {
        "n_sentences":   n_sents,
        "n_words":       n_words,
        "avg_sent_len":  round(asl, 1),
        "avg_word_len":  round(avg_word_len, 2),
        "avg_syllables": round(asw, 2),
        "flesch_ease":   round(fre, 1),
        "fk_grade":      round(fkgl, 1),
        "pct_complex":   round(pct_complex, 1),
        "ttr":           round(ttr, 2),
    }


def fre_label(score: float) -> str:
    if score >= 90: return "Very Easy (5th grade)"
    if score >= 80: return "Easy (6th grade)"
    if score >= 70: return "Fairly Easy (7th grade)"
    if score >= 60: return "Standard (8th-9th grade)"
    if score >= 50: return "Fairly Difficult (10-12th grade)"
    if score >= 30: return "Difficult (College level)"
    return "Very Difficult (Professional)"


# ─── SAMPLE TEXTS ─────────────────────────────────────────────────────────────

SAMPLES = {
    "Children's Story": (
        "The little rabbit hopped to the garden. She found a big red apple. "
        "She ate it and felt very happy. Then she ran home to her family. "
        "Her mother gave her a warm hug. It was a very good day."
    ),
    "News Article": (
        "City officials announced plans to expand the public transport network "
        "following a significant rise in commuter numbers. The new metro line "
        "will connect the eastern suburbs to the city centre by 2027. "
        "Construction is expected to create approximately 2,000 jobs. "
        "Critics have raised concerns about noise and disruption during the build."
    ),
    "Academic Abstract": (
        "This study investigates the neuropsychological correlates of working memory "
        "capacity in adolescent populations exposed to chronic socioeconomic stressors. "
        "Participants underwent functional magnetic resonance imaging while performing "
        "n-back tasks of varying cognitive load. Preliminary analyses indicate "
        "significant differences in prefrontal cortex activation patterns between "
        "high- and low-stress groups, suggesting that chronic adversity may modulate "
        "executive function through alterations in corticolimbic connectivity."
    ),
    "Legal Contract": (
        "Notwithstanding any provisions to the contrary contained herein, the Indemnifying "
        "Party shall defend, indemnify, and hold harmless the Indemnified Party from and "
        "against any and all claims, liabilities, obligations, damages, losses, costs, "
        "expenses, penalties, fines, and disbursements of any kind whatsoever arising "
        "out of or in connection with any breach of the representations, warranties, "
        "covenants, or obligations of the Indemnifying Party set forth in this Agreement."
    ),
}


# ─── DEMO ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print()
    print("=" * 70)
    print("  LANGUAGE COMPLEXITY ANALYZER DEMO")
    print("=" * 70)
    print("  Method: Flesch Reading Ease + Flesch-Kincaid Grade Level (rule-based)")
    print()

    all_metrics = {}
    for name, text in SAMPLES.items():
        m = analyse_text(text)
        all_metrics[name] = m

    # ── Per-sample report ─────────────────────────────────────────────────────
    for name, m in all_metrics.items():
        print(f"  SAMPLE: {name}")
        print(f"  " + "-" * 58)
        print(f"    Words          : {m['n_words']}  |  Sentences: {m['n_sentences']}")
        print(f"    Avg sent length: {m['avg_sent_len']:.1f} words  |  Avg word length: {m['avg_word_len']:.2f} chars")
        print(f"    Avg syllables  : {m['avg_syllables']:.2f} per word  |  Complex words (3+ syl): {m['pct_complex']:.1f}%")
        print(f"    Flesch Ease    : {m['flesch_ease']:.1f}  -> {fre_label(m['flesch_ease'])}")
        print(f"    FK Grade Level : {m['fk_grade']:.1f}  (US school grade equivalent)")
        print(f"    Lexical Diversity (TTR): {m['ttr']:.2f}  (unique/total words; higher=more varied)")
        print()

    # ── Comparative table ─────────────────────────────────────────────────────
    print("  COMPARISON TABLE:")
    print("  " + "-" * 70)
    print(f"  {'Sample':<22}  {'Ease':>6}  {'Grade':>6}  {'Avg Sent':>8}  {'Avg Syl':>8}  {'Complex%':>9}")
    print("  " + "-" * 70)
    for name, m in all_metrics.items():
        short = name[:21]
        print(f"  {short:<22}  {m['flesch_ease']:>6.1f}  {m['fk_grade']:>6.1f}  "
              f"{m['avg_sent_len']:>8.1f}  {m['avg_syllables']:>8.2f}  {m['pct_complex']:>9.1f}%")
    print("  " + "-" * 70)

    # ── Explanation ───────────────────────────────────────────────────────────
    print()
    print("  FORMULA EXPLANATION:")
    print("  Flesch Ease = 206.835 - 1.015*ASL - 84.6*ASW")
    print("    ASL = avg sentence length (words), ASW = avg syllables per word")
    print("  Score 90-100: very easy (children's stories)")
    print("  Score 60-70 : standard (newspapers, general fiction)")
    print("  Score 0-30  : very difficult (legal, academic, medical)")
    print()
    print("  Flesch-Kincaid Grade = 0.39*ASL + 11.8*ASW - 15.59")
    print("    Output: US school grade (e.g. 12 = senior high school, 16 = college)")
    print()
    print("[DONE] language_complexity_analyzer.py complete")

"""
Text Summarizer
================
What it does:
  Summarizes a long article two ways:
    (a) Extractive - selects the most important sentences using TF-IDF
        word-frequency scoring (no model download needed, always works)
    (b) Abstractive - uses HuggingFace's 'sshleifer/distilbart-cnn-6-6'
        (~300MB) if available; falls back to extractive if not downloaded
        yet or on first run with no internet.

What it teaches:
  - Extractive summarization: how TF-IDF scores rank sentences by
    informativeness without any model at all
  - Abstractive summarization: a seq2seq model rewrites the content
  - The key difference: extractive preserves original wording,
    abstractive generates new fluent sentences

How to run:
  python NLP\text_summarizer.py    (from PYTHON\ folder)

Estimated RAM: ~50MB (extractive only) | ~600MB (if abstractive model loads)
Time: <1s extractive | ~60s first run to download abstractive model

Model note: Extractive uses TF-IDF (classical, offline, always works).
Abstractive uses 'sshleifer/distilbart-cnn-6-6' wrapped in try/except;
if that fails the script still completes successfully with extractive only.
No API key needed.
"""

import re
import math
from collections import Counter

# ─── EXTRACTIVE SUMMARIZER ────────────────────────────────────────────────────

def tokenize(text):
    return re.findall(r"\b[a-z]{3,}\b", text.lower())

STOPWORDS = {
    "the","and","for","are","was","were","that","this","with","have","from",
    "they","been","its","also","but","not","can","has","all","into","which",
    "through","these","those","such","more","most","some","each","both",
    "their","when","where","what","who","how","then","than","over","about",
    "after","before","between","during","every","many","other","than","will",
}

def extractive_summary(text, n_sentences=3):
    """
    Score each sentence by the sum of TF-IDF weights of its words.
    Pick the top-N sentences in original order.
    """
    # Split into sentences (basic: split on . ! ?)
    sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if len(s.strip()) > 20]
    if len(sents) <= n_sentences:
        return " ".join(sents)

    # Build corpus-level word frequency (as proxy for IDF)
    all_words = [w for s in sents for w in tokenize(s) if w not in STOPWORDS]
    doc_count = len(sents)
    word_in_sent = Counter()  # how many sentences contain each word
    for s in sents:
        for w in set(tokenize(s)):
            if w not in STOPWORDS:
                word_in_sent[w] += 1

    # IDF: log(N / df)
    idf = {w: math.log(doc_count / (df + 1)) for w, df in word_in_sent.items()}

    scores = []
    for s in sents:
        words  = [w for w in tokenize(s) if w not in STOPWORDS]
        tf     = Counter(words)
        score  = sum(tf[w] * idf.get(w, 0) for w in tf) / max(len(words), 1)
        scores.append(score)

    # Pick top-N indices; preserve original order
    ranked   = sorted(range(len(sents)), key=lambda i: scores[i], reverse=True)
    top_idx  = sorted(ranked[:n_sentences])
    return " ".join(sents[i] for i in top_idx)

# ─── ABSTRACTIVE SUMMARIZER (with fallback) ───────────────────────────────────

def abstractive_summary(text, max_len=120, min_len=40):
    """
    Attempt to use HuggingFace distilbart summarizer.
    Falls back gracefully to None if the model isn't cached.
    """
    try:
        from transformers import pipeline
        summarizer = pipeline(
            "summarization",
            model="sshleifer/distilbart-cnn-6-6",
            device=-1,          # CPU
        )
        result = summarizer(text[:1024], max_length=max_len, min_length=min_len, do_sample=False)
        return result[0]["summary_text"].strip()
    except Exception as e:
        return None  # Caller handles fallback

# ─── DEMO ─────────────────────────────────────────────────────────────────────

ARTICLE = """
Artificial intelligence has rapidly transformed the way we interact with
technology over the past decade. Machine learning algorithms now power
recommendation systems on major streaming platforms, enabling users to
discover content tailored to their preferences. Natural language processing
advances have made virtual assistants more conversational and helpful.
Self-driving car research has progressed significantly, with several
companies conducting public road tests in controlled environments.

In healthcare, AI-assisted diagnostic tools can detect certain cancers
in medical imaging with accuracy comparable to experienced radiologists.
Drug discovery timelines have shortened as AI models predict molecular
interactions before expensive lab tests are run. However, experts caution
that AI systems must be rigorously validated before deployment in
high-stakes settings. Bias in training data can lead to unequal outcomes
for different patient populations.

The economic impact of AI is significant. Automation has changed the
nature of many jobs, displacing routine tasks while creating demand for
new skills such as data labeling, model evaluation, and AI ethics review.
Governments around the world are drafting AI regulations to ensure
transparency and accountability. The European Union's AI Act, for
example, categorizes AI applications by risk level and sets requirements
for documentation, testing, and human oversight.

Researchers continue to push the boundaries of what AI can do. Large
language models trained on vast corpora of text can write code, compose
essays, and answer complex questions. Yet these models sometimes produce
plausible-sounding but factually incorrect outputs, a problem known as
hallucination. Building reliable, trustworthy AI remains an open challenge
that the research community is actively working to address.
"""

if __name__ == "__main__":
    print("=" * 65)
    print("  TEXT SUMMARIZER DEMO")
    print("=" * 65)
    sent_pat = re.compile(r"(?<=[.!?])\s+")
    n_sents  = len(sent_pat.split(ARTICLE.strip()))
    print(f"\n  Article length: {len(ARTICLE.split())} words, {n_sents} sentences\n")

    # ── (a) Extractive ────────────────────────────────────────────────────────
    print("  (A) EXTRACTIVE SUMMARY (TF-IDF sentence scoring):")
    print("  " + "-" * 61)
    ext = extractive_summary(ARTICLE, n_sentences=3)
    for line in re.findall(r".{1,80}(?:\s|$)", ext):
        print(f"  {line.rstrip()}")
    print(f"\n  Method  : TF-IDF word scoring, no model, always offline")
    print(f"  Length  : {len(ext.split())} words (from {len(ARTICLE.split())} original)\n")

    # ── (b) Abstractive ───────────────────────────────────────────────────────
    print("  (B) ABSTRACTIVE SUMMARY (distilbart-cnn-6-6, if cached):")
    print("  " + "-" * 61)
    print("  Attempting to load sshleifer/distilbart-cnn-6-6 ...")
    abst = abstractive_summary(ARTICLE)
    if abst:
        for line in re.findall(r".{1,80}(?:\s|$)", abst):
            print(f"  {line.rstrip()}")
        print(f"\n  Method  : Abstractive seq2seq (distilbart), generates new sentences")
        print(f"  Length  : {len(abst.split())} words")
    else:
        print("  Model not cached / unavailable.")
        print("  Falling back to extractive (same as above).")
        print(f"  -> {ext[:200]}...")

    print("\n  COMPARISON:")
    print("  Extractive : preserves exact original sentences, very fast,")
    print("               100% offline, no model needed.")
    print("  Abstractive: generates fluent new sentences, may rephrase ideas,")
    print("               requires ~600MB model download on first use.")
    print("\n[DONE] text_summarizer.py complete")

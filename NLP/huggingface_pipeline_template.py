# NLP Reference — HuggingFace Pipeline Template (CPU-safe)
# Wrappers for common HF tasks with offline/mock fallbacks.
# ~300 MB RAM when loaded, <1s per inference on CPU (small models)
# NOTE: First run downloads model weights — requires internet once.

import os

# ─── GLOBAL CACHE ─────────────────────────────────────────────────────────────
_LOADED_PIPELINES = {}

def _get_pipeline(task: str, model: str, fallback_fn=None):
    """Load and cache a HuggingFace pipeline, with fallback."""
    key = f"{task}:{model}"
    if key in _LOADED_PIPELINES:
        return _LOADED_PIPELINES[key], "hf"
    try:
        from transformers import pipeline
        pipe = pipeline(task, model=model, device=-1)  # device=-1 = CPU
        _LOADED_PIPELINES[key] = pipe
        return pipe, "hf"
    except Exception as e:
        print(f"  [HF] Could not load {model} ({e}) — using fallback")
        return fallback_fn, "fallback"

# ─── 1. SENTIMENT ANALYSIS ────────────────────────────────────────────────────
def _sentiment_fallback(texts):
    """Lexicon-based fallback."""
    pos_words = {"good","great","excellent","love","amazing","best","happy","wonderful","fantastic"}
    neg_words = {"bad","terrible","awful","hate","poor","worst","horrible","sad","disappointing"}
    results = []
    for text in texts:
        words = set(text.lower().split())
        p = len(words & pos_words); n = len(words & neg_words)
        label = "POSITIVE" if p >= n else "NEGATIVE"
        score = min(1.0, 0.5 + abs(p-n)*0.1)
        results.append({"label": label, "score": round(score, 3)})
    return results

def sentiment_pipeline(texts: list, model="distilbert-base-uncased-finetuned-sst-2-english"):
    """
    Sentiment analysis. Returns list of {"label": "POSITIVE"/"NEGATIVE", "score": float}.
    Falls back to lexicon-based scorer if model unavailable.
    """
    if isinstance(texts, str): texts = [texts]
    pipe, ptype = _get_pipeline("sentiment-analysis", model, None)
    if ptype == "hf":
        return pipe(texts, truncation=True, max_length=128)
    return _sentiment_fallback(texts)

# ─── 2. ZERO-SHOT CLASSIFICATION ──────────────────────────────────────────────
def _zero_shot_fallback(texts, labels):
    """TF-IDF cosine similarity fallback for zero-shot."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    import numpy as np
    corpus = texts + labels
    vec  = TfidfVectorizer(max_features=500).fit_transform(corpus).toarray()
    t_vecs = vec[:len(texts)]
    l_vecs = vec[len(texts):]
    results = []
    for tv in t_vecs:
        sims = l_vecs.dot(tv) / (np.linalg.norm(l_vecs,axis=1)*np.linalg.norm(tv)+1e-9)
        best = int(np.argmax(sims))
        results.append({"sequence": "", "labels": labels,
                        "scores": sims.tolist(), "top_label": labels[best]})
    return results

def zero_shot_pipeline(texts, candidate_labels: list,
                        model="facebook/bart-large-mnli"):
    """
    Zero-shot classification. Returns dict with labels + scores.
    NOTE: bart-large-mnli is ~1.6GB — may fail on 8GB RAM.
    Falls back to TF-IDF similarity.
    """
    if isinstance(texts, str): texts = [texts]
    pipe, ptype = _get_pipeline("zero-shot-classification", model, None)
    if ptype == "hf":
        return [pipe(t, candidate_labels) for t in texts]
    return _zero_shot_fallback(texts, candidate_labels)

# ─── 3. NER ────────────────────────────────────────────────────────────────────
def _ner_fallback(texts):
    """Regex-based NER fallback."""
    import re
    results = []
    for text in texts:
        entities = []
        for m in re.finditer(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b", text):
            entities.append({"word": m.group(), "entity_group": "MISC",
                             "score": 0.7, "start": m.start(), "end": m.end()})
        results.append(entities)
    return results

def ner_pipeline(texts, model="dslim/bert-base-NER"):
    """Named Entity Recognition. Falls back to regex if model unavailable."""
    if isinstance(texts, str): texts = [texts]
    pipe, ptype = _get_pipeline("ner", model, None)
    if ptype == "hf":
        return [pipe(t) for t in texts]
    return _ner_fallback(texts)

# ─── 4. TEXT SUMMARIZATION ────────────────────────────────────────────────────
def _summarize_extractive(text: str, n_sentences=2) -> str:
    """Extractive summarization fallback (TF-IDF sentence scoring)."""
    import re
    import numpy as np
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    if len(sentences) <= n_sentences: return text
    from sklearn.feature_extraction.text import TfidfVectorizer
    tfidf = TfidfVectorizer().fit_transform(sentences).toarray()
    scores = tfidf.sum(axis=1)
    top_idx = sorted(np.argsort(scores)[-n_sentences:])
    return " ".join(sentences[i] for i in top_idx)

def summarize_pipeline(texts, min_length=20, max_length=80,
                        model="sshleifer/distilbart-cnn-12-6"):
    """
    Text summarization. Falls back to extractive summarizer.
    NOTE: distilbart is ~1.2GB — may fail on 8GB RAM.
    """
    if isinstance(texts, str): texts = [texts]
    pipe, ptype = _get_pipeline("summarization", model, None)
    if ptype == "hf":
        return [pipe(t, min_length=min_length, max_length=max_length,
                     truncation=True)[0]["summary_text"] for t in texts]
    return [_summarize_extractive(t) for t in texts]

# ─── 5. TEXT GENERATION ───────────────────────────────────────────────────────
def _generation_fallback(prompt: str, max_new_tokens=30) -> str:
    """Simple word-repeat / template fallback."""
    words = prompt.split()
    return prompt + " " + " ".join(words[:min(5, len(words))] * 3)[:max_new_tokens]

def generate_pipeline(prompt: str, max_new_tokens=50,
                       model="distilgpt2"):
    """Text generation with distilgpt2 (~85MB). Falls back to template."""
    pipe, ptype = _get_pipeline("text-generation", model, None)
    if ptype == "hf":
        out = pipe(prompt, max_new_tokens=max_new_tokens,
                   do_sample=True, temperature=0.8, pad_token_id=50256)
        return out[0]["generated_text"]
    return _generation_fallback(prompt, max_new_tokens)

# ─── DEMO ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=== Sentiment Analysis ===")
    texts = [
        "This movie was absolutely wonderful, I loved every minute!",
        "Terrible experience. The product broke after one day.",
        "It was okay, nothing special.",
    ]
    results = sentiment_pipeline(texts)
    for t, r in zip(texts, results):
        print(f"  [{r['label']:8s} {r['score']:.2f}] {t[:50]}")

    print("\n=== Extractive Summarization (fallback) ===")
    article = (
        "Machine learning is a type of artificial intelligence. "
        "It allows systems to learn from data automatically. "
        "Deep learning is a subset that uses neural networks. "
        "These networks have many layers of connected nodes. "
        "They excel at tasks like image recognition and NLP."
    )
    summary = _summarize_extractive(article, n_sentences=2)
    print(f"  Summary: {summary}")

    print("\nTemplate loaded. Import and call functions in your own scripts.")

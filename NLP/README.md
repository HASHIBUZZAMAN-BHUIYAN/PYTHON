# NLP Reference Folder

Reusable templates for Natural Language Processing tasks.

| File | Contents |
|------|----------|
| `text_preprocessing_template.py`  | Tokenization, stopwords, stemming, lemmatization, TF-IDF |
| `embedding_similarity_template.py`| Word2Vec training, document vectors, cosine similarity search |
| `huggingface_pipeline_template.py`| HuggingFace pipeline wrappers with CPU-safe fallbacks |

## Environment Setup

This folder has its **own dedicated virtual environment** (`NLP\.venv`) — separate from every other folder including BEGINNER and ADVANCED.

**From a fresh terminal:**
```
cd C:\Users\zen\Documents\GitHub\PYTHON
NLP\.venv\Scripts\activate
python NLP\text_summarizer.py
```

**Or:** double-click `NLP\activate.bat` — it activates the venv and sets the working directory automatically.

Installed packages (see `NLP\requirements.txt`): numpy, scikit-learn, scipy, matplotlib, gensim, nltk (+ punkt/stopwords data), sentence-transformers, transformers, torch (CPU)

Note: scripts that use sentence-transformers or transformers models (faq_chatbot, document_comparator, text_summarizer, etc.) download the model once on first run (~90-600 MB) and cache it in `~\.cache\huggingface\hub\`. Subsequent runs are instant.

---

## Related lessons
- ADVANCED/week4/day22 — Text preprocessing & TF-IDF
- ADVANCED/week4/day23 — Word embeddings
- ADVANCED/week4/day24 — Sentiment analysis
- ADVANCED/week4/day25 — NER & POS tagging
- ADVANCED/week4/day26 — Text classification
- ADVANCED/week4/day27 — Text generation & summarization
- ADVANCED/week4/day28 — Retrieval-based chatbot (capstone)

## Hardware note
All templates run on CPU. Large models (e.g., full BERT) require ~1-2 GB RAM.
Prefer `distilbert-base-uncased` (~260 MB) or `all-MiniLM-L6-v2` (~90 MB).

---

## Runnable Projects

All commands are run from the `PYTHON/` folder using the venv Python.
No API key required. Classical/rule-based methods always work; model-based
methods auto-download lightweight weights (<300MB) on first run.

### Practical Text Tools

| File | Description | Method | Run command |
|------|-------------|--------|-------------|
| `text_summarizer.py` | Extractive (TF-IDF) + optional abstractive summarization | TF-IDF (always); distilbart-cnn-6-6 if cached | `python NLP\text_summarizer.py` |
| `grammar_style_checker.py` | Passive voice, repeated words, long sentences, confusable pairs, intensifiers | 100% regex, no model | `python NLP\grammar_style_checker.py` |
| `paraphraser_and_simplifier.py` | Synonym-map word simplification + sentence splitting; optional T5 paraphrase | Synonym map (always); t5-small if cached | `python NLP\paraphraser_and_simplifier.py` |

### Chatbots

| File | Description | Method | Run command |
|------|-------------|--------|-------------|
| `faq_chatbot.py` | Retrieval FAQ bot matching questions to 10 built-in FAQs via embedding cosine similarity | all-MiniLM-L6-v2 (~90MB) | `python NLP\faq_chatbot.py` |
| `intent_based_bot.py` | 5-intent classifier (greeting/complaint/order_status/goodbye/small_talk) trained on 50 built-in examples | TF-IDF + LogisticRegression (offline) | `python NLP\intent_based_bot.py` |
| `multi_turn_context_bot.py` | Slot-filling chatbot: tracks name/topic/order across 10 demo turns via state machine | 100% rule-based, no model | `python NLP\multi_turn_context_bot.py` |

### Analysis Tools

| File | Description | Method | Run command |
|------|-------------|--------|-------------|
| `document_comparator.py` | Pairwise similarity of 4 docs: TF-IDF (lexical) vs embeddings (semantic) — explains the difference | TF-IDF + all-MiniLM-L6-v2 | `python NLP\document_comparator.py` |
| `topic_extractor.py` | Topic discovery on 8 docs via LDA and TF-IDF keyword clustering; compares both methods | sklearn LDA + TF-IDF (offline) | `python NLP\topic_extractor.py` |
| `review_sentiment_dashboard.py` | Sentiment dashboard for 12 reviews: scores, % breakdown, top keywords, urgent-flag | Lexicon (always); distilbert-sst-2 if cached | `python NLP\review_sentiment_dashboard.py` |
| `language_complexity_analyzer.py` | Flesch Reading Ease + FK Grade Level for 4 samples (children's to legal); all arithmetic | 100% formula-based, no model | `python NLP\language_complexity_analyzer.py` |

> **Hardware:** All projects run CPU-only. Peak RAM ~300MB (embedding model).
> Slowest run: ~2s for FAQ chatbot encoding. All data is built-in — no datasets to download.

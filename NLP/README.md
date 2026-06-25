# NLP Reference Folder

Reusable templates for Natural Language Processing tasks.

| File | Contents |
|------|----------|
| `text_preprocessing_template.py`  | Tokenization, stopwords, stemming, lemmatization, TF-IDF |
| `embedding_similarity_template.py`| Word2Vec training, document vectors, cosine similarity search |
| `huggingface_pipeline_template.py`| HuggingFace pipeline wrappers with CPU-safe fallbacks |

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

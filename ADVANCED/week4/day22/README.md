# Day 22: Text Preprocessing

## Overview
Text preprocessing is the foundation of every NLP pipeline. Raw text must be cleaned, normalized, and transformed into numeric representations before any model can learn from it.

## Topics Covered
- Tokenization (split-based and NLTK word_tokenize)
- Stopwords removal with NLTK
- Stemming with PorterStemmer vs Lemmatization with WordNetLemmatizer
- Regex-based text cleaning
- Bag-of-Words from scratch
- TF-IDF from scratch AND with sklearn CountVectorizer / TfidfVectorizer

## Learning Objectives
By the end of this day you will be able to:
1. Clean and normalize raw text for NLP tasks
2. Explain the difference between stemming and lemmatization
3. Build a Bag-of-Words representation by hand
4. Compute TF-IDF scores both manually and with sklearn
5. Use TF-IDF features with a simple classifier

## Files
| File | Purpose |
|------|---------|
| `lesson.py` | Complete walkthrough of all preprocessing concepts |
| `exercises.py` | 5 practice exercises (TODO stubs) |
| `solutions.py` | Full working solutions to all 5 exercises |
| `projects/project1_spam_classifier.py` | TF-IDF + Logistic Regression spam classifier |
| `projects/project2_keyword_extractor.py` | Top-K keyword extractor using TF-IDF |
| `projects/project3_plagiarism_checker.py` | Cosine similarity plagiarism checker |

## Setup
```
pip install nltk scikit-learn
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('punkt_tab')"
```

## Hardware Notes
- CPU-only, no GPU required
- Peak RAM: ~150 MB
- Estimated runtime: ~10 seconds

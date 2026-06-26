# Advanced Day 25 — NER & POS Tagging
# ~50 MB RAM, ~3s on CPU

print("""
=== NER & POS Tagging — Day 25 ===

POS tagging assigns grammatical categories to each word (noun, verb, adj…).
NER identifies "real-world" entity spans: people, places, organizations, dates.

Approaches:
  1. Rule-based regex — fast, no download, handles known patterns
  2. NLTK chunking  — uses MaxEnt POS tagger + regex grammar rules
  3. spaCy en_core_web_sm (~12MB) — statistical model, state-of-the-art accuracy
""")

TEXT = (
    "Elon Musk founded SpaceX in 2002 in Hawthorne, California. "
    "The company secured a $1.6 billion contract from NASA in March 2006. "
    "Jeff Bezos also operates Blue Origin, based in Kent, Washington. "
    "In 2023, OpenAI released GPT-4 which partnered with Microsoft in Redmond."
)

# ─── 1. REGEX NER (no dependencies) ──────────────────────────────────────────
print("=== 1. Regex NER ===")
import re

ENTITY_PATTERNS = {
    "MONEY": r"\$[\d,]+(?:\.\d+)?(?:\s+(?:billion|million|thousand))?",
    "DATE":  r"\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|"
             r"Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|"
             r"Dec(?:ember)?)\s+\d{4}\b|\b\d{4}\b",
    "ORG":   r"\b(?:SpaceX|NASA|Blue Origin|OpenAI|Microsoft|Google|Apple|Amazon)\b",
    "GPE":   r"\b(?:California|Washington|Hawthorne|Kent|Redmond|Seattle|New York)\b",
    "PERSON":r"\b(?:Elon Musk|Jeff Bezos|Sam Altman|Satya Nadella)\b",
}

for label, pattern in ENTITY_PATTERNS.items():
    matches = re.findall(pattern, TEXT)
    if matches:
        print(f"  {label:8s}: {matches}")

# ─── 2. NLTK POS + NE CHUNKING ────────────────────────────────────────────────
print("\n=== 2. NLTK POS Tagging ===")
try:
    import nltk
    for pkg in ["punkt","punkt_tab","averaged_perceptron_tagger",
                "averaged_perceptron_tagger_eng","maxent_ne_chunker",
                "maxent_ne_chunker_tab","words"]:
        nltk.download(pkg, quiet=True)

    tokens = nltk.word_tokenize(TEXT)
    pos_tags = nltk.pos_tag(tokens)
    print("  First 15 tokens with POS tags:")
    for word, tag in pos_tags[:15]:
        print(f"    {word:<20} {tag}")

    from collections import Counter
    tag_dist = Counter(tag for _, tag in pos_tags)
    top_tags = tag_dist.most_common(6)
    print(f"\n  Top POS tags: {top_tags}")

except Exception as e:
    print(f"  NLTK unavailable ({type(e).__name__}) — skip")

# ─── 3. SPACY NER ────────────────────────────────────────────────────────────
print("\n=== 3. spaCy NER ===")
try:
    import spacy
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(TEXT)

    print("  Entities found:")
    for ent in doc.ents:
        print(f"    [{ent.label_:8s}] '{ent.text}'  (char {ent.start_char}–{ent.end_char})")

    print("\n  POS tags (first 12 tokens):")
    for tok in list(doc)[:12]:
        print(f"    {tok.text:<20} POS={tok.pos_:<8} TAG={tok.tag_:<8} DEP={tok.dep_}")

except Exception as e:
    print(f"  spaCy unavailable ({type(e).__name__}) — using regex fallback")
    for label, pattern in ENTITY_PATTERNS.items():
        matches = re.findall(pattern, TEXT)
        if matches:
            print(f"    [{label:8s}] {matches}")

print("""
=== Key Takeaways ===
  POS tags: NN(noun) VB(verb) JJ(adjective) RB(adverb) IN(preposition) DT(determiner)
  NER entity types: PERSON, ORG, GPE(geo-political), DATE, MONEY, PRODUCT
  spaCy > NLTK for accuracy; regex for known-pattern extraction with no model
""")

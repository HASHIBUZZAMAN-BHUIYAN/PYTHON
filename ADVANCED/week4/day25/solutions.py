# Advanced Day 25 — Solutions
import re, nltk
from collections import Counter
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

for pkg in ["punkt","punkt_tab","averaged_perceptron_tagger","averaged_perceptron_tagger_eng"]:
    nltk.download(pkg, quiet=True)

TEXT = (
    "Elon Musk founded SpaceX in 2002 in Hawthorne, California. "
    "The company secured a $1.6 billion contract from NASA in March 2006. "
    "Jeff Bezos also operates Blue Origin, based in Kent, Washington. "
    "In 2023, OpenAI released GPT-4 which partnered with Microsoft in Redmond."
)

# Ex 1
print("=== Ex 1: Email & Phone NER ===")
PATTERNS = {
    "EMAIL": r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+",
    "PHONE": r"\+?[\d][\d\-\s.]{8,14}[\d]",
    "ORG":   r"\b(?:SpaceX|NASA|Blue Origin|OpenAI|Microsoft)\b",
}
test = "Contact john.doe@company.com or call +1-555-867-5309 for info."
for label, pat in PATTERNS.items():
    m = re.findall(pat, test)
    if m: print(f"  {label}: {m}")

# Ex 2
print("\n=== Ex 2: POS Frequency Chart ===")
para = (
    "The quick brown fox jumps over the lazy dog. "
    "Scientists discovered a new planet orbiting the distant star. "
    "Beautiful flowers bloom rapidly in warm spring weather. "
    "She quietly reads complex novels every evening before sleeping. "
    "Modern computers process data faster than traditional machines."
)
tokens   = nltk.word_tokenize(para)
pos_tags = nltk.pos_tag(tokens)
SIMPLIFY = {"NN":"NOUN","NNS":"NOUN","NNP":"NOUN","NNPS":"NOUN",
            "VB":"VERB","VBD":"VERB","VBG":"VERB","VBN":"VERB","VBP":"VERB","VBZ":"VERB",
            "JJ":"ADJ","JJR":"ADJ","JJS":"ADJ",
            "RB":"ADV","RBR":"ADV","RBS":"ADV"}
counts = Counter(SIMPLIFY.get(t,"OTHER") for _,t in pos_tags)
cats   = ["NOUN","VERB","ADJ","ADV","OTHER"]
vals   = [counts.get(c,0) for c in cats]
fig,ax = plt.subplots(figsize=(7,3))
ax.bar(cats, vals, color=["steelblue","tomato","limegreen","orange","grey"])
ax.set_title("POS Distribution"); ax.set_ylabel("Count")
plt.tight_layout(); plt.savefig("pos_dist.png",dpi=80); plt.close()
print(f"  {dict(zip(cats,vals))}  → saved pos_dist.png")

# Ex 3
print("\n=== Ex 3: Entity Co-occurrence ===")
ENTITY_PATS = {
    "PERSON":r"\b(?:Elon Musk|Jeff Bezos|Sam Altman)\b",
    "ORG":   r"\b(?:SpaceX|NASA|Blue Origin|OpenAI|Microsoft)\b",
    "GPE":   r"\b(?:California|Washington|Hawthorne|Kent|Redmond)\b",
    "MONEY": r"\$[\d,]+(?:\.\d+)?(?:\s+billion)?",
    "DATE":  r"\b\d{4}\b",
}
sentences = re.split(r"(?<=[.!?]) ", TEXT)
cooc = {(a,b):0 for a in ENTITY_PATS for b in ENTITY_PATS if a<b}
for sent in sentences:
    found = [lbl for lbl, pat in ENTITY_PATS.items() if re.search(pat, sent)]
    for i,a in enumerate(found):
        for b in found[i+1:]:
            pair = (min(a,b),max(a,b))
            if pair in cooc: cooc[pair]+=1
print("  Co-occurrence counts:")
for pair, cnt in cooc.items():
    if cnt: print(f"    {pair[0]:8s} ↔ {pair[1]:8s}: {cnt}")

# Ex 4
print("\n=== Ex 4: Proper Noun Extraction ===")
news = (
    "Tim Cook announced the new Apple iPhone in Cupertino last week. "
    "Google DeepMind released AlphaFold in London. "
    "Satya Nadella met Bill Gates at the Microsoft campus in Seattle."
)
tok = nltk.word_tokenize(news)
pos = nltk.pos_tag(tok)
# Gather consecutive NNP/NNPS sequences
entities, buf = [], []
for word, tag in pos + [("", "END")]:
    if tag in ("NNP","NNPS"):
        buf.append(word)
    else:
        if buf: entities.append(" ".join(buf))
        buf = []
print(f"  Proper noun sequences: {entities}")

# Ex 5
print("\n=== Ex 5: Regex vs spaCy comparison ===")
regex_ents = set()
for label, pat in ENTITY_PATS.items():
    for m in re.findall(pat, TEXT):
        regex_ents.add((label, m))
try:
    import spacy
    nlp   = spacy.load("en_core_web_sm")
    spacy_ents = {(e.label_, e.text) for e in nlp(TEXT).ents}
    regex_only = regex_ents - spacy_ents
    spacy_only = spacy_ents - regex_ents
    both       = regex_ents & spacy_ents
    print(f"  Regex only : {regex_only}")
    print(f"  spaCy only : {spacy_only}")
    print(f"  Both found : {both}")
except:
    print("  spaCy unavailable — regex results:")
    for e in sorted(regex_ents): print(f"    {e}")

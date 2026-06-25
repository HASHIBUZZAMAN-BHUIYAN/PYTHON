"""
Project: News Article NER Analyzer
Teaches: extracting who/what/where/when from a news article, frequency
         ranking of entities, visualization as bar chart.
~30 MB RAM, ~2s on CPU
"""
import re
from collections import Counter
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ARTICLE = """
Apple Inc. announced a partnership with OpenAI in Cupertino, California on June 10, 2024.
Tim Cook, the CEO of Apple, confirmed that GPT-4 capabilities will be integrated into
the iPhone 16 lineup starting September 2024. Microsoft also holds a $13 billion investment
in OpenAI and was represented by Satya Nadella at the San Francisco press conference.
The deal follows Google's announcement in February 2024 to integrate Gemini AI into Android.
Amazon Web Services meanwhile expanded its data centers in Dublin, Ireland and Singapore
during Q1 2024, investing $5.3 billion globally. Jeff Bezos and Elon Musk both commented
on the rapid pace of AI integration at the World Economic Forum in Davos, Switzerland.
Tesla reported a 20% revenue increase in Q2 2024 with headquarters in Austin, Texas.
The European Union passed the AI Act in Brussels, Belgium with effective date March 2024.
"""

ENTITY_PATTERNS = {
    "PERSON": r"\b(?:Tim Cook|Satya Nadella|Jeff Bezos|Elon Musk|Sam Altman|Sundar Pichai)\b",
    "ORG":    r"\b(?:Apple(?:\s+Inc\.?)?|OpenAI|Microsoft|Google|Amazon(?:\s+Web\s+Services)?|"
              r"Tesla|European Union|AWS)\b",
    "GPE":    r"\b(?:California|Cupertino|San Francisco|Dublin|Ireland|Singapore|Texas|Austin|"
              r"Brussels|Belgium|Davos|Switzerland)\b",
    "MONEY":  r"\$[\d,]+(?:\.\d+)?\s*billion",
    "DATE":   r"\b(?:January|February|March|April|May|June|July|August|September|"
              r"October|November|December)\s+\d{4}\b|\bQ[1-4]\s+\d{4}\b|\b\d{4}\b",
    "PRODUCT":r"\b(?:iPhone\s+16|GPT-4|Gemini AI|AI Act)\b",
}

print("=== News NER Analyzer ===\n")

all_entities = {}
for label, pattern in ENTITY_PATTERNS.items():
    matches = re.findall(pattern, ARTICLE)
    all_entities[label] = matches
    if matches:
        freq = Counter(matches)
        print(f"{label} ({len(matches)} total):")
        for entity, count in freq.most_common(5):
            print(f"  {entity:<30} ×{count}")
        print()

# ─── Summary stats ────────────────────────────────────────────────────────────
total_entities = sum(len(v) for v in all_entities.values())
print(f"Total entities extracted: {total_entities}")

# ─── Visualization ────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(14, 7))
axes = axes.flatten()

for idx, (label, entities) in enumerate(all_entities.items()):
    if not entities:
        axes[idx].set_visible(False)
        continue
    freq   = Counter(entities).most_common(5)
    names  = [e[0][:20] for e in freq]
    counts = [e[1] for e in freq]
    axes[idx].barh(names[::-1], counts[::-1], color=f"C{idx}", alpha=0.8, edgecolor="k", linewidth=0.5)
    axes[idx].set_title(f"{label} ({len(entities)} mentions)")
    axes[idx].set_xlabel("Count")

plt.suptitle("News Article — Entity Frequency by Type", fontsize=12, y=1.01)
plt.tight_layout(); plt.savefig("news_ner.png", dpi=85, bbox_inches="tight"); plt.close()
print("Saved news_ner.png")

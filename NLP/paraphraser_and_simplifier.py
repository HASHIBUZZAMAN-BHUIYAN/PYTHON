"""
Paraphraser and Simplifier
===========================
What it does:
  Takes sample sentences/paragraphs and produces:
    (a) Simplified version: replaces complex words using a synonym map,
        splits long sentences at conjunctions, shortens words where possible.
        100% rule-based, fully offline, no model needed.
    (b) Paraphrase via T5 model ('t5-small') if it is already cached on
        this machine — otherwise falls back gracefully to (a) only.

What it teaches:
  - Rule-based text simplification: how synonym lookup + sentence splitting
    reduces complexity without any ML at all
  - Readability trade-offs: shorter words and sentences are easier to read
    but may lose nuance
  - How seq2seq models (T5) can reformulate text vs rule-based approaches
  - Graceful fallback pattern: always produce a useful result even when a
    model is unavailable

How to run:
  python NLP\paraphraser_and_simplifier.py    (from PYTHON\ folder)

Estimated RAM: ~50MB (rule-based only) | ~300MB (if T5 model loads)
Time: <1s rule-based | ~30s first run for T5 model download

Model note: Rule-based simplification always runs (no model). T5-small
paraphrasing wrapped in try/except — script completes successfully with
rule-based only if T5 is not cached. No API key needed.
"""

import re

# ─── SYNONYM MAP (complex -> simple) ──────────────────────────────────────────
# Covers academic/legal/technical words commonly swapped in plain-English guides

SIMPLIFY_MAP = {
    "utilize":       "use",
    "utilise":       "use",
    "approximately": "about",
    "commence":      "start",
    "terminate":     "end",
    "endeavour":     "try",
    "endeavor":      "try",
    "sufficient":    "enough",
    "demonstrate":   "show",
    "obtain":        "get",
    "purchase":      "buy",
    "require":       "need",
    "assist":        "help",
    "assistance":    "help",
    "subsequently":  "then",
    "prior to":      "before",
    "in order to":   "to",
    "due to the fact that": "because",
    "at this point in time": "now",
    "in the event that": "if",
    "for the purpose of": "to",
    "with regard to": "about",
    "in accordance with": "following",
    "notwithstanding": "despite",
    "nevertheless":  "still",
    "aforementioned": "earlier",
    "aforementioned": "earlier",
    "facilitate":    "help",
    "implement":     "do",
    "modification":  "change",
    "modifications": "changes",
    "substantial":   "large",
    "additional":    "more",
    "component":     "part",
    "components":    "parts",
    "numerous":      "many",
    "individuals":   "people",
    "residence":     "home",
    "occupation":    "job",
    "comprehend":    "understand",
    "inquire":       "ask",
    "respond":       "answer",
    "provide":       "give",
    "notify":        "tell",
    "determine":     "find out",
    "approximately": "about",
    "initiate":      "start",
    "complete":      "finish",
}

# Sorted by length (longest first) so multi-word phrases match before single words
SIMPLIFY_MAP_SORTED = sorted(SIMPLIFY_MAP.items(), key=lambda x: -len(x[0]))


def simplify_words(text: str) -> str:
    """Replace complex words/phrases from SIMPLIFY_MAP.
    Uses word boundaries (\b) for single-word entries so 'implement' never
    mangles 'implementation', 'require' never produces 'needd', etc.
    Multi-word phrases use their natural whitespace boundaries.
    """
    result = text
    for complex_word, simple_word in SIMPLIFY_MAP_SORTED:
        escaped = re.escape(complex_word)
        # For single words, add word boundaries to avoid partial matches inside longer words
        if " " not in complex_word:
            pat_str = r"\b" + escaped + r"\b"
        else:
            pat_str = escaped  # multi-word phrases: whitespace already anchors them
        pattern = re.compile(pat_str, re.IGNORECASE)
        def replacer(m, sw=simple_word):
            if m.group(0)[0].isupper():
                return sw[0].upper() + sw[1:]
            return sw
        result = pattern.sub(replacer, result)
    return result


def split_long_sentences(text: str, max_words: int = 20) -> str:
    """
    Split sentences longer than max_words at 'and', 'but', 'because', 'which',
    'that', or 'while' — inserting a period and capitalising the next word.
    """
    SPLIT_WORDS = ["and", "but", "because", "which", "while", "although"]
    sents = re.split(r"(?<=[.!?])\s+", text.strip())
    out   = []
    for sent in sents:
        words = sent.split()
        if len(words) <= max_words:
            out.append(sent)
            continue
        # Find the best split point near the middle
        mid   = len(words) // 2
        best  = -1
        for i in range(mid - 5, mid + 6):
            if 0 < i < len(words) and words[i].lower() in SPLIT_WORDS:
                best = i
                break
        if best == -1:
            # No conjunction near middle — just keep the long sentence
            out.append(sent)
        else:
            part1 = " ".join(words[:best]).rstrip(",")
            part2_words = words[best:]
            part2_words[0] = part2_words[0].capitalize()
            part2 = " ".join(part2_words)
            # Add period if part1 doesn't end in punctuation
            if not part1[-1:] in ".!?":
                part1 += "."
            out.append(part1)
            out.append(part2)
    return " ".join(out)


def rule_based_simplify(text: str) -> str:
    """Apply word simplification then sentence splitting."""
    text = simplify_words(text)
    text = split_long_sentences(text)
    return text


# ─── T5-BASED PARAPHRASE (with fallback) ─────────────────────────────────────

def t5_paraphrase(text: str) -> str | None:
    """
    Try to paraphrase using t5-small (a lightweight seq2seq model).
    Prepend 'paraphrase: ' as the task prefix.
    Returns None if the model is not cached or fails to load.
    """
    try:
        from transformers import T5ForConditionalGeneration, T5Tokenizer
        tokenizer = T5Tokenizer.from_pretrained("t5-small", legacy=False)
        model     = T5ForConditionalGeneration.from_pretrained("t5-small")
        model.eval()

        input_ids = tokenizer(
            "paraphrase: " + text,
            return_tensors="pt", max_length=256, truncation=True
        ).input_ids

        import torch
        with torch.no_grad():
            outputs = model.generate(
                input_ids,
                max_length=200, num_beams=4,
                early_stopping=True,
            )
        return tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
    except Exception:
        return None


# ─── DEMO ─────────────────────────────────────────────────────────────────────

SAMPLES = [
    (
        "Academic / bureaucratic",
        "The committee will endeavour to utilize all available resources "
        "in order to facilitate the implementation of the aforementioned "
        "modifications to the curriculum prior to the commencement of "
        "the subsequent academic year, notwithstanding any challenges "
        "that may arise during this process.",
    ),
    (
        "Legal / formal",
        "In accordance with the provisions outlined in the agreement, "
        "all individuals residing at the aforementioned residence are "
        "required to provide sufficient assistance to the authorities "
        "due to the fact that an investigation has been initiated.",
    ),
    (
        "Technical documentation",
        "The system will subsequently determine whether it has obtained "
        "sufficient data to complete the calibration process and will "
        "notify the operator in the event that additional components "
        "require modification.",
    ),
]

if __name__ == "__main__":
    print()
    print("=" * 65)
    print("  PARAPHRASER & SIMPLIFIER DEMO")
    print("=" * 65)

    print("\n  Checking for T5-small model (optional, only if cached)...")
    t5_available = False
    try:
        import os
        hf_cache = os.path.expanduser("~/.cache/huggingface/hub")
        t5_dirs  = [d for d in (os.listdir(hf_cache) if os.path.isdir(hf_cache) else [])
                    if "t5-small" in d]
        t5_available = bool(t5_dirs)
    except Exception:
        pass
    print(f"  T5-small cached: {'YES' if t5_available else 'NO (will use rule-based only)'}")

    for label, text in SAMPLES:
        print()
        print("=" * 65)
        print(f"  SAMPLE: {label}")
        print("-" * 65)
        wrapped = re.sub(r"(.{1,72})(\s|$)", r"  ORIGINAL  | \1\n", text).strip()
        print(f"  ORIGINAL  | {text[:72]}")
        if len(text) > 72:
            for chunk in re.findall(r".{1,72}", text[72:]):
                print(f"            | {chunk}")

        simplified = rule_based_simplify(text)
        print(f"\n  (A) SIMPLIFIED (rule-based):")
        for chunk in re.findall(r".{1,65}", simplified):
            print(f"      {chunk}")

        orig_words  = len(text.split())
        simpl_words = len(simplified.split())
        print(f"      [Word count: {orig_words} -> {simpl_words}, "
              f"reduction: {(orig_words-simpl_words)/orig_words*100:.0f}%]")

        if t5_available:
            print(f"\n  (B) PARAPHRASE (T5-small, loading...):")
            para = t5_paraphrase(text)
            if para:
                for chunk in re.findall(r".{1,65}", para):
                    print(f"      {chunk}")
            else:
                print("      T5 failed — showing rule-based only.")
        else:
            print(f"\n  (B) PARAPHRASE: T5-small not cached. Run once online")
            print(f"      to download (~240MB); then this section auto-enables.")

    print()
    print("  COMPARISON NOTE:")
    print("  Rule-based: Fast, 100% offline, predictable substitutions.")
    print("  T5 model  : Rewrites structure, not just words. Requires ~300MB")
    print("              RAM and ~30s download on first run.")
    print()
    print("[DONE] paraphraser_and_simplifier.py complete")

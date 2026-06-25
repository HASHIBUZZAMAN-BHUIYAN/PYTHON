"""
Project: Structured Story Generator
Teaches: combining Markov chains with templates and slot-filling to generate
         coherent short stories with a defined 3-act structure.
~10 MB RAM, <1s on CPU
"""
import re, random
from collections import defaultdict

# ─── Character & setting vocabularies ────────────────────────────────────────
CHARACTERS = ["an old lighthouse keeper", "a young scientist", "a brave explorer",
              "a wise wizard", "a curious robot", "a lost astronaut"]
SETTINGS   = ["a hidden island", "a dark forest", "an ancient city",
              "a distant planet", "a stormy sea", "a quiet village"]
OBJECTS    = ["a golden key", "a mysterious map", "a glowing crystal",
              "an ancient book", "a broken compass", "a strange device"]
CHALLENGES = ["could not find the way back", "discovered a dangerous secret",
              "had to solve an impossible puzzle", "was chased by unknown creatures",
              "found the world was not what it seemed"]
RESOLUTIONS= ["succeeded after great courage and clever thinking",
              "found an unexpected ally who helped solve the mystery",
              "discovered the answer had been within all along",
              "outsmarted the danger with patience and wisdom",
              "returned changed forever but wiser"]

# ─── Sentence-level Markov model ──────────────────────────────────────────────
STORY_CORPUS = """
The wind howled through the dark trees as the traveler walked slowly.
Suddenly a bright light appeared on the distant horizon ahead.
The old stone path led deeper into the mysterious unknown forest.
Strange sounds echoed through the cold empty corridors of the castle.
The brave hero had faced many dangers before this final challenge.
A warm fire flickered casting dancing shadows on the rough stone walls.
The stars above seemed brighter than they had ever seemed before.
Rain fell softly on the ancient cobblestone streets of the old city.
The morning mist slowly lifted revealing a breathtaking stunning view.
""".strip()

tokens = re.sub(r"[^a-z\s]", "", STORY_CORPUS.lower()).split()
bigram = defaultdict(list)
for i in range(len(tokens)-1):
    bigram[tokens[i]].append(tokens[i+1])

def gen_sentence(start="the", n=10, seed=0):
    random.seed(seed)
    word = start; result = [word]
    for _ in range(n-1):
        nxt = bigram.get(word, list(bigram.keys()))
        word = random.choice(nxt)
        result.append(word)
    return " ".join(result).capitalize() + "."

# ─── Story generator ──────────────────────────────────────────────────────────
def generate_story(seed=42):
    random.seed(seed)
    char  = random.choice(CHARACTERS)
    place = random.choice(SETTINGS)
    obj   = random.choice(OBJECTS)
    chal  = random.choice(CHALLENGES)
    resol = random.choice(RESOLUTIONS)

    # Act 1: Setup
    act1_env = gen_sentence("the", n=12, seed=seed)
    act1 = (f"Once upon a time, {char} arrived at {place}. "
            f"{act1_env} "
            f"While exploring, they discovered {obj} hidden beneath the ground.")

    # Act 2: Conflict
    act2_env = gen_sentence("strange", n=12, seed=seed+1)
    act2 = (f"Things quickly became difficult. {char.capitalize()} {chal}. "
            f"{act2_env} "
            f"The situation seemed impossible and hopeless.")

    # Act 3: Resolution
    act3_env = gen_sentence("the", n=12, seed=seed+2)
    act3 = (f"But in the end, {char} {resol}. "
            f"{act3_env} "
            f"With {obj} secured safely, they left {place} forever changed.")

    return act1, act2, act3

print("=== Structured Story Generator ===\n")
for seed in [0, 7, 42]:
    act1, act2, act3 = generate_story(seed=seed)
    print(f"{'─'*60}")
    print(f"Story #{seed+1}")
    print(f"\n[Act 1 — Setup]\n  {act1}")
    print(f"\n[Act 2 — Conflict]\n  {act2}")
    print(f"\n[Act 3 — Resolution]\n  {act3}\n")

print("Each story uses random slot-filling for structure + Markov for atmospheric sentences.")

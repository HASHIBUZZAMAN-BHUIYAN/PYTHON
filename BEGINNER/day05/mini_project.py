# Day 05 Mini-Project — Text Analyzer
# Analyzes a block of text entered by the user.

print("=== Text Analyzer ===")
print("Enter your text (blank line to finish):\n")
lines = []
while True:
    line = input()
    if line == "":
        break
    lines.append(line)

text = "\n".join(lines)
words = text.split()
sentences = text.count('.') + text.count('!') + text.count('?')
chars_no_space = len(text.replace(" ", "").replace("\n", ""))

freq = {}
for w in words:
    clean = w.strip(".,!?;:\"'").lower()
    if clean:
        freq[clean] = freq.get(clean, 0) + 1

top5 = sorted(freq.items(), key=lambda x: x[1], reverse=True)[:5]

print("\n" + "=" * 40)
print("ANALYSIS RESULTS")
print("=" * 40)
print(f"Characters (with spaces) : {len(text)}")
print(f"Characters (no spaces)   : {chars_no_space}")
print(f"Words                    : {len(words)}")
print(f"Unique words             : {len(freq)}")
print(f"Sentences (approx.)      : {sentences}")
print(f"\nTop 5 words:")
for word, count in top5:
    print(f"  '{word}': {count}")

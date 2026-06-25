# Day 03 Mini-Project — Grade Report
# Asks for up to 5 subject scores and prints a full grade report.

subjects = ["Math", "Science", "English", "History", "Art"]
scores = []

print("=== Grade Report Generator ===")
for subject in subjects:
    raw = input(f"{subject} score (0-100, or press Enter to skip): ")
    if raw.strip() == "":
        continue
    scores.append((subject, float(raw)))

if not scores:
    print("No scores entered.")
else:
    total = sum(s for _, s in scores)
    avg   = total / len(scores)

    print("\n" + "-" * 35)
    print(f"{'Subject':<12} {'Score':>6} {'Grade':>6}")
    print("-" * 35)
    for subject, s in scores:
        if s >= 90:   g = "A"
        elif s >= 80: g = "B"
        elif s >= 70: g = "C"
        elif s >= 60: g = "D"
        else:         g = "F"
        print(f"{subject:<12} {s:>6.1f} {g:>6}")
    print("-" * 35)
    print(f"{'Average':<12} {avg:>6.1f}")
    overall = "Pass" if avg >= 60 else "Fail"
    print(f"\nOverall result: {overall}")

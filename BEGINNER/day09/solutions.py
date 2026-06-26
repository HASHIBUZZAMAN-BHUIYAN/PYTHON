# Day 09 — Solutions
import csv, os

# Exercise 1
def count_lines(filepath):
    with open(filepath, encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())

# Create test file
with open("test_lines.txt", "w") as f:
    f.write("hello\n\nworld\n\ngoodbye\n")
print(count_lines("test_lines.txt"))
os.remove("test_lines.txt")

# Exercise 2
def word_freq(filepath):
    freq = {}
    punct = ".,!?;:\"'()[]"
    with open(filepath, encoding="utf-8") as f:
        for line in f:
            for word in line.split():
                w = word.strip(punct).lower()
                if w:
                    freq[w] = freq.get(w, 0) + 1
    return freq

# Exercise 3 & 4
rows = [
    ["Name","Subject","Score"],
    ["Alice","Math",90],["Alice","Science",75],
    ["Bob","Math",65],["Bob","Science",88],
    ["Carol","Math",82],["Carol","Science",95],
]
with open("scores.csv","w",newline="") as f:
    csv.writer(f).writerows(rows)

print("Scores > 80:")
subject_scores = {}
with open("scores.csv") as f:
    for row in csv.DictReader(f):
        s = int(row["Score"])
        if s > 80:
            print(f"  {row['Name']} {row['Subject']} {s}")
        subject_scores.setdefault(row["Subject"], []).append(s)

print("\nAverage per subject:")
for subj, scores in subject_scores.items():
    print(f"  {subj}: {sum(scores)/len(scores):.1f}")

os.remove("scores.csv")

# Exercise 5
def save_todos(todos, filepath):
    with open(filepath, "w", encoding="utf-8") as f:
        f.writelines(t + "\n" for t in todos)

def load_todos(filepath):
    if not os.path.exists(filepath): return []
    with open(filepath, encoding="utf-8") as f:
        return [l.rstrip() for l in f if l.strip()]

save_todos(["Buy milk", "Write code", "Exercise"], "todos.txt")
todos = load_todos("todos.txt")
print(todos)
os.remove("todos.txt")

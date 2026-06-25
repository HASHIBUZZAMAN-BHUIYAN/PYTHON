# Day 06 Mini-Project — Student Score Manager
# Add/view/analyze student scores stored in lists.

students = []
scores   = []

def print_menu():
    print("\n1. Add student  2. View all  3. Stats  4. Top N  5. Quit")

while True:
    print_menu()
    choice = input("Choice: ").strip()
    if choice == "1":
        name  = input("Student name: ")
        score = float(input("Score: "))
        students.append(name)
        scores.append(score)
        print(f"Added {name} with score {score}.")
    elif choice == "2":
        if not students:
            print("No students yet.")
        else:
            print(f"\n{'Name':<15} {'Score':>6}")
            print("-" * 22)
            for n, s in sorted(zip(students, scores), key=lambda x: x[1], reverse=True):
                print(f"{n:<15} {s:>6.1f}")
    elif choice == "3":
        if scores:
            s = sorted(scores)
            n = len(s)
            med = (s[n//2-1]+s[n//2])/2 if n%2==0 else s[n//2]
            print(f"Count={n}  Mean={sum(s)/n:.1f}  Median={med:.1f}  Min={min(s)}  Max={max(s)}")
    elif choice == "4":
        k = int(input("Show top N: "))
        pairs = sorted(zip(students, scores), key=lambda x: x[1], reverse=True)[:k]
        for i, (n, s) in enumerate(pairs, 1):
            print(f"  {i}. {n} — {s}")
    elif choice == "5":
        print("Bye!")
        break
    else:
        print("Invalid choice.")

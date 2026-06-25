# Day 02 Mini-Project — Simple Bill Splitter
# Calculates how to split a restaurant bill among friends.

print("=== Bill Splitter ===\n")
bill      = float(input("Total bill amount ($): "))
tip_pct   = float(input("Tip percentage (e.g. 15): "))
people    = int(input("Number of people: "))

tip_amount    = bill * tip_pct / 100
total         = bill + tip_amount
per_person    = total / people

print(f"\nBill       : ${bill:.2f}")
print(f"Tip ({tip_pct:.0f}%)  : ${tip_amount:.2f}")
print(f"Total      : ${total:.2f}")
print(f"Per person : ${per_person:.2f}")

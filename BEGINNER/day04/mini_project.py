# Day 04 Mini-Project — Multiplication Table Printer
# Prints an N×N multiplication table with aligned columns.

n = int(input("Table size (e.g. 10): "))
width = len(str(n * n)) + 1   # column width based on largest number

# Header row
print(" " * width, end="")
for col in range(1, n + 1):
    print(f"{col:{width}}", end="")
print()
print("-" * (width * (n + 1)))

# Table rows
for row in range(1, n + 1):
    print(f"{row:{width}}", end="")
    for col in range(1, n + 1):
        print(f"{row*col:{width}}", end="")
    print()

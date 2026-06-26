# Day 10 — Exercises

# Exercise 1 — Robust input
# Write get_int(prompt) that keeps asking the user until they enter a valid int.
# Use a while loop + try/except.

# Exercise 2 — Safe file reader
# Write safe_read(filepath) that returns the file content as a list of lines,
# or an empty list if the file doesn't exist (no crash).

# Exercise 3 — Custom exceptions
# Define a ValidationError exception.
# Write validate_username(name) that raises ValidationError if:
# - length < 3 or > 20
# - contains spaces
# - starts with a digit
# Otherwise returns True. Test all cases.

# Exercise 4 — Context manager with exception
# Write divide_all(numbers, divisor) that divides every number in the list
# by divisor. If divisor is 0, raise a ZeroDivisionError with a helpful message.
# Use try/except in the caller to print the error and continue gracefully.

# Exercise 5 — Chained exceptions
# Python lets you raise one exception from another:
#   raise RuntimeError("wrapper") from original_exc
# Write a function that reads a CSV number from a string, but if parsing fails,
# raises a descriptive RuntimeError chained from the original ValueError.

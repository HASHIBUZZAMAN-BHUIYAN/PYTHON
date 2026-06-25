# string_tools.py — custom module for Day 13 Exercise 5

def reverse_words(s):
    return " ".join(s.split()[::-1])

def is_anagram(a, b):
    clean = lambda s: sorted(s.lower().replace(" ", ""))
    return clean(a) == clean(b)

def word_count(s):
    return len(s.split())

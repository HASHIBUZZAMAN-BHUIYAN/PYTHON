# ─────────────────────────────────────────────
#  SECTION 6: User Input (interactive)
# ─────────────────────────────────────────────

print("\n=== User Input ===")

user_name  = input("Enter your name: ")
birth_year = int(input("Enter your birth year: "))

current_year   = 2026
age_calculated = current_year - birth_year
days_alive     = age_calculated * 365

print(f"\nWelcome, {user_name}!")
print(f"You are approximately {age_calculated} years old.")
print(f"That's roughly {days_alive:,} days of experience — now add Python!")

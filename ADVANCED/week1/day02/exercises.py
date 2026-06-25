# Advanced Day 02 — Exercises
import pandas as pd, numpy as np

np.random.seed(1)

# Shared dataset for exercises 1-4
sales = pd.DataFrame({
    "date":     pd.date_range("2024-01-01", periods=100, freq="D"),
    "product":  np.random.choice(["Widget","Gadget","Doohickey"], 100),
    "region":   np.random.choice(["North","South","East","West"], 100),
    "units":    np.random.randint(1, 50, 100),
    "price":    np.random.choice([9.99, 19.99, 4.99], 100),
})
sales["revenue"] = sales["units"] * sales["price"]

# Exercise 1 — Filtering & derived columns
# Add a column "is_high_value" = True if revenue > 200.
# Print the top 5 high-value sales sorted by revenue.
# TODO

# Exercise 2 — GroupBy
# Find total revenue per product.
# Find average units sold per region.
# Which region has the highest average revenue?
# TODO

# Exercise 3 — Missing data handling
df_m = pd.DataFrame({
    "A": [1, None, 3, None, 5],
    "B": [None, 2, None, 4, 5],
    "C": ["x","y",None,"w",None]
})
# a) Print count of nulls per column
# b) Fill numeric nulls with column median
# c) Fill string nulls with "unknown"
# d) Drop rows that STILL have any null
# TODO

# Exercise 4 — Merge & pivot
dept = pd.DataFrame({
    "product": ["Widget","Gadget","Doohickey"],
    "category": ["Hardware","Electronics","Misc"]
})
# Merge sales with dept on "product".
# Then build a pivot table: rows=region, cols=category, values=revenue (sum).
# TODO

# Exercise 5 — String operations on a column
# Given the DataFrame below, clean the 'email' column:
# lowercase, strip spaces, mark invalid (no '@') as NaN.
people = pd.DataFrame({
    "name":  ["Alice","Bob","Carol"],
    "email": ["  Alice@Example.COM ", "bobATgmail.com", "carol@test.org"]
})
# TODO

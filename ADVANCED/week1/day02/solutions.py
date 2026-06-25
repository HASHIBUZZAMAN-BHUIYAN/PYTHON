# Advanced Day 02 — Solutions
import pandas as pd, numpy as np
np.random.seed(1)

sales = pd.DataFrame({
    "date":    pd.date_range("2024-01-01", periods=100, freq="D"),
    "product": np.random.choice(["Widget","Gadget","Doohickey"], 100),
    "region":  np.random.choice(["North","South","East","West"], 100),
    "units":   np.random.randint(1, 50, 100),
    "price":   np.random.choice([9.99, 19.99, 4.99], 100),
})
sales["revenue"] = sales["units"] * sales["price"]

# Exercise 1
sales["is_high_value"] = sales["revenue"] > 200
print(sales[sales["is_high_value"]].sort_values("revenue", ascending=False).head(5))

# Exercise 2
print(sales.groupby("product")["revenue"].sum())
print(sales.groupby("region")["units"].mean())
print(sales.groupby("region")["revenue"].mean().idxmax(), "has highest avg revenue")

# Exercise 3
df_m = pd.DataFrame({"A":[1,None,3,None,5],"B":[None,2,None,4,5],"C":["x","y",None,"w",None]})
print(df_m.isnull().sum())
for col in ["A","B"]:
    df_m[col] = df_m[col].fillna(df_m[col].median())
df_m["C"] = df_m["C"].fillna("unknown")
print(df_m.dropna())

# Exercise 4
dept = pd.DataFrame({"product":["Widget","Gadget","Doohickey"],"category":["Hardware","Electronics","Misc"]})
merged = pd.merge(sales, dept, on="product")
pivot = pd.pivot_table(merged, values="revenue", index="region", columns="category", aggfunc="sum").fillna(0)
print(pivot)

# Exercise 5
people = pd.DataFrame({"name":["Alice","Bob","Carol"],"email":["  Alice@Example.COM ","bobATgmail.com","carol@test.org"]})
people["email"] = people["email"].str.strip().str.lower()
people.loc[~people["email"].str.contains("@"), "email"] = float("nan")
print(people)

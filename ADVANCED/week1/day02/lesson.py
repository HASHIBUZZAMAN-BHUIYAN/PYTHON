# Advanced Day 02 — Pandas Deep Dive
# ~100 MB RAM, <5s on CPU

import pandas as pd
import numpy as np

np.random.seed(42)

# ─── 1. SERIES ───────────────────────────────────────────────────────────────
s = pd.Series([10, 20, 30, 40], index=["a","b","c","d"])
print(s)
print(s["b"])         # 20
print(s[s > 15])      # boolean indexing
print(s.mean(), s.std())

# ─── 2. DATAFRAME ────────────────────────────────────────────────────────────
data = {
    "name":   ["Alice","Bob","Carol","Dave","Eve"],
    "age":    [25, 30, 28, 35, 22],
    "city":   ["Dhaka","London","Dhaka","Tokyo","London"],
    "salary": [50000, 70000, 55000, 90000, 45000],
    "score":  [88, 75, 92, 67, 95],
}
df = pd.DataFrame(data)
print(df)
print(df.head(3))
print(df.tail(2))
print(df.info())
print(df.describe())

# ─── 3. SELECTING DATA ───────────────────────────────────────────────────────
print(df["name"])               # single column (Series)
print(df[["name","salary"]])    # multiple columns (DataFrame)
print(df.loc[0])                # row by label
print(df.iloc[2])               # row by position
print(df.loc[1:3, ["name","score"]])  # rows 1-3, 2 cols

# ─── 4. FILTERING ────────────────────────────────────────────────────────────
print(df[df["age"] > 27])
print(df[(df["city"] == "Dhaka") & (df["score"] >= 85)])
print(df[df["city"].isin(["London","Tokyo"])])

# ─── 5. ADDING / MODIFYING COLUMNS ───────────────────────────────────────────
df["bonus"] = df["salary"] * 0.10
df["grade"] = df["score"].apply(lambda s: "A" if s>=90 else ("B" if s>=80 else "C"))
print(df)

# ─── 6. DROPPING ─────────────────────────────────────────────────────────────
df2 = df.drop(columns=["bonus"])
df3 = df.drop(index=[0,1])
print(df2.columns.tolist())

# ─── 7. SORTING ──────────────────────────────────────────────────────────────
print(df.sort_values("salary", ascending=False))
print(df.sort_values(["city","score"], ascending=[True,False]))

# ─── 8. HANDLING MISSING DATA ────────────────────────────────────────────────
df_missing = pd.DataFrame({"x":[1,None,3,None,5], "y":[10,20,None,40,50]})
print(df_missing.isnull().sum())
print(df_missing.dropna())
print(df_missing.fillna(df_missing.mean(numeric_only=True)))

# ─── 9. GROUPBY ──────────────────────────────────────────────────────────────
grouped = df.groupby("city")["salary"].agg(["mean","min","max","count"])
print(grouped)

print(df.groupby("city").agg(
    avg_salary=("salary","mean"),
    avg_score=("score","mean"),
    count=("name","count")
))

# ─── 10. MERGE / JOIN ────────────────────────────────────────────────────────
dept = pd.DataFrame({
    "name": ["Alice","Bob","Carol","Dave","Eve"],
    "dept": ["Eng","HR","Eng","Sales","HR"]
})
merged = pd.merge(df[["name","salary"]], dept, on="name", how="left")
print(merged)

# ─── 11. PIVOT TABLE ─────────────────────────────────────────────────────────
print(pd.pivot_table(df, values="salary", index="city",
                     aggfunc=["mean","count"]))

# Advanced Day 01 — Solutions
import numpy as np
np.random.seed(0)

# Exercise 1
m = np.zeros((5,5), dtype=int)
m[0,:] = m[-1,:] = m[:,0] = m[:,-1] = 1
print(m)

# Exercise 2
arr = np.random.rand(100)
norm = (arr - arr.mean()) / arr.std()
print(f"Mean: {norm.mean():.6f}, Std: {norm.std():.6f}")

# Exercise 3
a = np.array([1,2,3]).reshape(3,1)
b = np.array([4,5,6,7]).reshape(1,4)
outer = a * b
print(outer)

# Exercise 4
M = np.random.randint(1, 101, size=(6,4))
print("Matrix:\n", M)
row_means = M.mean(axis=1)
print("Row with highest mean:", np.argmax(row_means), row_means[np.argmax(row_means)])
col_stds = M.std(axis=0)
print("Col with lowest std:", np.argmin(col_stds), col_stds[np.argmin(col_stds)])
print("Overall median:", np.median(M))

# Exercise 5
x = np.array([1,2,3,4,5,6,7,8], dtype=float)
y = np.array([2.1,3.9,6.2,7.8,10.1,12.0,14.2,15.9])
X = np.column_stack([x, np.ones_like(x)])
coeffs = np.linalg.inv(X.T @ X) @ X.T @ y
m, b = coeffs
print(f"Slope m={m:.4f}, Intercept b={b:.4f}")
print("Predictions:", (X @ coeffs).round(2))

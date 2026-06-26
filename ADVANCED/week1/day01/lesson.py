# Advanced Day 01 — NumPy Deep Dive
# ~50 MB RAM, <5s on CPU

import numpy as np

# ─── 1. CREATING ARRAYS ──────────────────────────────────────────────────────
a = np.array([1, 2, 3, 4, 5])
b = np.array([[1, 2, 3], [4, 5, 6]])

print(a.shape, a.dtype)          # (5,) int64
print(b.shape, b.ndim, b.size)   # (2,3) 2  6

print(np.zeros((3, 3)))
print(np.ones((2, 4), dtype=float))
print(np.eye(3))                  # identity matrix
print(np.arange(0, 10, 2))       # [0 2 4 6 8]
print(np.linspace(0, 1, 5))      # [0. .25 .5 .75 1.]
print(np.random.seed(42); np.random.rand(2, 3))

# ─── 2. INDEXING & SLICING ───────────────────────────────────────────────────
m = np.arange(12).reshape(3, 4)
print(m)
print(m[1, 2])        # row 1, col 2 → 6
print(m[0, :])        # first row
print(m[:, 1])        # second column
print(m[1:, 2:])      # sub-matrix

print(m[m > 7])       # all elements > 7

rows = np.array([0, 2])
print(m[rows, :])     # rows 0 and 2

# ─── 3. VIEWS vs COPIES ─────────────────────────────────────────────────────
v = m[0, :]           # VIEW — shares memory with m
v[0] = 999
print(m[0])           # m is modified!

c = m[0, :].copy()    # COPY — independent
c[0] = 0
print(m[0])           # m unchanged

# ─── 4. ARITHMETIC & BROADCASTING ────────────────────────────────────────────
x = np.array([1, 2, 3])
print(x + 10)         # [11 12 13]  — scalar broadcast
print(x * 2)
print(x ** 2)

a = np.array([[1], [2], [3]])    # shape (3,1)
b = np.array([10, 20, 30, 40])   # shape (4,)
print(a + b)                     # shape (3,4) — broadcast

A = np.array([[1,2],[3,4]])
B = np.array([[5,6],[7,8]])
print(A * B)           # element-wise
print(A @ B)           # matrix multiplication
print(np.dot(A, B))    # same as @

# ─── 5. AGGREGATIONS ─────────────────────────────────────────────────────────
data = np.random.randint(1, 100, size=(4, 5))
print(data)
print(data.sum())          # total
print(data.sum(axis=0))    # column sums
print(data.sum(axis=1))    # row sums
print(data.mean(), data.std(), data.min(), data.max())
print(np.percentile(data, [25, 50, 75]))

# ─── 6. UNIVERSAL FUNCTIONS (ufuncs) ─────────────────────────────────────────
angles = np.linspace(0, 2*np.pi, 5)
print(np.sin(angles).round(2))
print(np.exp(np.array([0, 1, 2])))   # e^x
print(np.log(np.array([1, np.e, np.e**2])))

# ─── 7. RESHAPING ────────────────────────────────────────────────────────────
flat = np.arange(24)
print(flat.reshape(2, 3, 4))   # 3D
print(flat.reshape(6, -1))     # -1 inferred = 4

# ─── 8. LINEAR ALGEBRA ───────────────────────────────────────────────────────
A = np.array([[2., 1.], [1., 3.]])

print("Determinant:", np.linalg.det(A))
print("Inverse:\n",    np.linalg.inv(A))
print("Eigenvalues:",  np.linalg.eig(A)[0])

b = np.array([8., 13.])
x = np.linalg.solve(A, b)
print("Solution x:", x)         # [3. 2.]
print("Verify Ax:", A @ x)      # [8. 13.]

# ─── 9. STACKING & SPLITTING ─────────────────────────────────────────────────
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
print(np.vstack([a, b]))        # (2,3)
print(np.hstack([a, b]))        # (6,)
print(np.concatenate([a, b]))   # same as hstack for 1D

parts = np.split(np.arange(12), 3)
print(parts)

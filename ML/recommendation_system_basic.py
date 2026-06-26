"""
What it does: Builds a user-based collaborative filtering recommender on a synthetic
              user-item rating matrix (30 users x 20 movies).
What it teaches: Cosine similarity from scratch, sparse matrix handling, top-N
                 recommendation logic, rating matrix heatmap visualization.
Category: RECOMMENDATION SYSTEMS
RAM estimate: < 50 MB
Time estimate: < 3 seconds
"""

import os
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.makedirs("ML/outputs", exist_ok=True)

np.random.seed(42)
n_users  = 30
n_movies = 20

ratings_raw = np.random.choice([0, 1, 2, 3, 4, 5],
                               size=(n_users, n_movies),
                               p=[0.40, 0.05, 0.10, 0.15, 0.20, 0.10])

movie_names = [f"Movie_{i+1:02d}" for i in range(n_movies)]
user_names  = [f"User_{i+1:02d}"  for i in range(n_users)]
R = pd.DataFrame(ratings_raw, index=user_names, columns=movie_names)

print(f"Rating matrix: {R.shape}  |  Non-zero entries: {(R > 0).sum().sum()}")
print(f"Sparsity: {(R == 0).sum().sum() / R.size:.1%}")


def cosine_sim(a, b):
    """Cosine similarity between two rating vectors (ignore zeros = unrated)."""
    mask = (a > 0) & (b > 0)
    if mask.sum() == 0:
        return 0.0
    a_m = a[mask].astype(float)
    b_m = b[mask].astype(float)
    denom = np.linalg.norm(a_m) * np.linalg.norm(b_m)
    if denom == 0:
        return 0.0
    return float(np.dot(a_m, b_m) / denom)


def user_similarities(R_matrix, target_idx):
    """Return cosine similarities between target user and all others."""
    target_vec = R_matrix.iloc[target_idx].values
    sims = []
    for i in range(len(R_matrix)):
        if i == target_idx:
            sims.append(-1.0)  # exclude self
        else:
            sims.append(cosine_sim(target_vec, R_matrix.iloc[i].values))
    return np.array(sims)


def recommend(R_matrix, target_user, top_n_users=3, top_n_items=5):
    """
    For target_user:
      1. Find top_n_users most similar users.
      2. Aggregate their ratings (weighted by similarity) for unrated movies.
      3. Return top_n_items recommendations.
    """
    target_idx   = R_matrix.index.get_loc(target_user)
    target_rated = R_matrix.loc[target_user]
    unrated_mask = target_rated == 0

    sims       = user_similarities(R_matrix, target_idx)
    top_idxs   = np.argsort(sims)[::-1][:top_n_users]
    top_sims   = sims[top_idxs]
    top_users  = [R_matrix.index[i] for i in top_idxs]

    print(f"\nTop-{top_n_users} similar users to {target_user}:")
    for u, s in zip(top_users, top_sims):
        print(f"  {u}  similarity={s:.4f}")

    # Weighted average of neighbor ratings for unrated movies
    scores = {}
    for movie in R_matrix.columns[unrated_mask]:
        neighbor_ratings = R_matrix.loc[top_users, movie].values.astype(float)
        valid = neighbor_ratings > 0
        if valid.sum() == 0:
            continue
        weighted_sum = (neighbor_ratings[valid] * top_sims[valid]).sum()
        sim_sum      = top_sims[valid].sum()
        scores[movie] = weighted_sum / sim_sum if sim_sum > 0 else 0.0

    if not scores:
        print("  No recommendations possible (neighbors have no ratings for unrated movies).")
        return []

    sorted_recs = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_n_items]
    print(f"\nTop-{top_n_items} recommendations for {target_user}:")
    print("-" * 40)
    for rank, (movie, score) in enumerate(sorted_recs, 1):
        print(f"  {rank}. {movie}  predicted score={score:.2f}")
    return sorted_recs


recs = recommend(R, target_user="User_01", top_n_users=3, top_n_items=5)

fig, ax = plt.subplots(figsize=(10, 8))
masked = np.ma.masked_where(ratings_raw == 0, ratings_raw.astype(float))
cmap = plt.cm.YlOrRd
cmap.set_bad(color="lightgrey")
im = ax.imshow(masked, aspect="auto", cmap=cmap, vmin=1, vmax=5)
plt.colorbar(im, ax=ax, label="Rating (0=unrated=grey)")
ax.set_xticks(range(n_movies))
ax.set_xticklabels(movie_names, rotation=45, ha="right", fontsize=7)
ax.set_yticks(range(n_users))
ax.set_yticklabels(user_names, fontsize=7)
ax.set_title("User-Item Rating Matrix (grey = unrated)")
plt.tight_layout()
plt.savefig("ML/outputs/recommendation_heatmap.png", dpi=100)
plt.close()
print("\nPlot saved -> ML/outputs/recommendation_heatmap.png")

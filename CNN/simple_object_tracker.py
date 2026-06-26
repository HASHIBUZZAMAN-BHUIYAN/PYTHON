"""
Simple Object Tracker (Centroid Tracking)
==========================================
What it does:
  Generates a sequence of 10 synthetic frames where 2 coloured shapes
  (one circle, one square) move across the frame with constant velocity
  plus random jitter. Then:
    1. Detects each object per frame using contour-finding (classical CV)
    2. Links detections across frames by minimum centroid distance
       (centroid tracking -- the same approach used in SORT, DeepSORT)
    3. Assigns a persistent ID to each track across all frames
    4. Plots a grid of all 10 frames with bounding boxes + track IDs
    5. Plots the full trajectory of each tracked object

  This is the SAME conceptual pipeline as real multi-object trackers
  (SORT, DeepSORT, ByteTrack) -- they replace the contour detector with
  a YOLO CNN, and optionally add a re-ID embedding network.

What it teaches:
  - Tracking = detection + association across time
  - Centroid tracking: greedy assignment by nearest neighbour distance
  - Track lifecycle: born when a new detection appears, killed when a
    detection is missing for too many frames (max_age parameter)
  - How DeepSORT adds an appearance embedding to handle ID switches when
    objects cross or occlude each other

How to run:
  python CNN\simple_object_tracker.py   (from PYTHON\ folder)

Estimated RAM: <100MB | Time: <5s (no CNN training, pure CV)
Model: no neural network -- contour detection (OpenCV) + centroid matching.
Output: CNN\outputs\object_tracker.png
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import cv2

os.makedirs("CNN/outputs", exist_ok=True)
rng = np.random.RandomState(42)

IMG_SIZE  = 160
N_FRAMES  = 10
SHAPE_R   = 16   # radius / half-size of tracked shapes

# ─── SYNTHETIC SEQUENCE GENERATOR ─────────────────────────────────────────────

def make_sequence(n_frames=N_FRAMES):
    """
    Generate n_frames images with 2 moving objects.
    Object 0: blue circle, starts top-left, moves right+down
    Object 1: red square, starts top-right, moves left+down
    Returns: list of (frame_img, [box0, box1]) where box = (x1,y1,x2,y2)
    """
    frames = []
    # Starting positions
    pos = np.array([[15, 15], [IMG_SIZE - 15, 15]], dtype=float)
    vel = np.array([[10, 7], [-9, 8]], dtype=float)
    colours = [(40, 80, 220), (220, 50, 50)]   # BGR for cv2
    shape_types = ["circle", "square"]

    for t in range(n_frames):
        frame = np.full((IMG_SIZE, IMG_SIZE, 3), 245, dtype=np.uint8)
        boxes = []
        for i in range(2):
            # Bounce off edges
            for d in range(2):
                if pos[i, d] < SHAPE_R or pos[i, d] > IMG_SIZE - SHAPE_R:
                    vel[i, d] *= -1
                pos[i, d] = np.clip(pos[i, d], SHAPE_R, IMG_SIZE - SHAPE_R)

            cx, cy = int(pos[i, 0]), int(pos[i, 1])
            col    = colours[i]
            if shape_types[i] == "circle":
                cv2.circle(frame, (cx, cy), SHAPE_R, col, -1)
            else:
                cv2.rectangle(frame, (cx - SHAPE_R, cy - SHAPE_R),
                              (cx + SHAPE_R, cy + SHAPE_R), col, -1)
            boxes.append((cx - SHAPE_R, cy - SHAPE_R,
                          cx + SHAPE_R, cy + SHAPE_R))
            pos[i] += vel[i] + rng.normal(0, 1.5, 2)

        frames.append((frame, boxes))
    return frames


# ─── CENTROID DETECTOR ────────────────────────────────────────────────────────

def detect_centroids(frame):
    """
    Detect objects by colour thresholding + contour finding.
    Returns list of (cx, cy, x1, y1, x2, y2) for each detected object.
    """
    # Convert to HSV for colour thresholding
    hsv    = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
    detections = []

    # Blue object: H in [90,130] (OpenCV H range 0-179)
    blue_mask  = cv2.inRange(hsv, (100, 80, 80), (130, 255, 255))
    # Red object: H near 0 or 179
    red_mask_1 = cv2.inRange(hsv, (0,  80, 80), (10,  255, 255))
    red_mask_2 = cv2.inRange(hsv, (165, 80, 80), (179, 255, 255))
    red_mask   = cv2.bitwise_or(red_mask_1, red_mask_2)

    for mask in [blue_mask, red_mask]:
        mask   = cv2.dilate(mask, None, iterations=2)
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in cnts:
            if cv2.contourArea(cnt) < 100:
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            cx, cy = x + w // 2, y + h // 2
            detections.append((cx, cy, x, y, x + w, y + h))
    return detections


# ─── CENTROID TRACKER ─────────────────────────────────────────────────────────

class CentroidTracker:
    """
    Tracks objects by minimum Euclidean distance between frame centroids.
    Each track has an ID and a history of centroid positions.
    """

    def __init__(self, max_distance=50, max_age=3):
        self.max_distance = max_distance   # max pixels to match
        self.max_age      = max_age        # frames until track is dropped
        self.next_id      = 0
        self.tracks       = {}   # id -> {"centroid": (cx,cy), "age":int, "hist": list}

    def update(self, detections):
        """
        detections: list of (cx, cy, x1, y1, x2, y2)
        Returns: list of (track_id, cx, cy, x1, y1, x2, y2)
        """
        if not detections:
            # Age all tracks; remove if too old
            to_del = []
            for tid in self.tracks:
                self.tracks[tid]["age"] += 1
                if self.tracks[tid]["age"] > self.max_age:
                    to_del.append(tid)
            for tid in to_del:
                del self.tracks[tid]
            return []

        # Greedy nearest-neighbour matching
        unmatched_dets = list(range(len(detections)))
        result = []

        if self.tracks:
            track_ids   = list(self.tracks.keys())
            track_cents = np.array([self.tracks[tid]["centroid"] for tid in track_ids])
            det_cents   = np.array([(d[0], d[1]) for d in detections])

            # Distance matrix
            dists = np.linalg.norm(
                track_cents[:, None, :] - det_cents[None, :, :], axis=2
            )  # shape (n_tracks, n_dets)

            matched_tracks = set()
            matched_dets   = set()
            # Sort by distance, assign greedily
            for _ in range(min(len(track_ids), len(detections))):
                if dists.size == 0:
                    break
                row, col = np.unravel_index(dists.argmin(), dists.shape)
                if dists[row, col] > self.max_distance:
                    break
                if row in matched_tracks or col in matched_dets:
                    dists[row, col] = 1e9
                    continue
                tid = track_ids[row]
                cx, cy, x1, y1, x2, y2 = detections[col]
                self.tracks[tid]["centroid"] = (cx, cy)
                self.tracks[tid]["age"]      = 0
                self.tracks[tid]["hist"].append((cx, cy))
                result.append((tid, cx, cy, x1, y1, x2, y2))
                matched_tracks.add(row)
                matched_dets.add(col)
                dists[row, col] = 1e9
            unmatched_dets = [i for i in range(len(detections)) if i not in matched_dets]

        # Create new tracks for unmatched detections
        for det_idx in unmatched_dets:
            cx, cy, x1, y1, x2, y2 = detections[det_idx]
            tid = self.next_id
            self.next_id += 1
            self.tracks[tid] = {"centroid": (cx, cy), "age": 0, "hist": [(cx, cy)]}
            result.append((tid, cx, cy, x1, y1, x2, y2))

        # Age + prune unmatched tracks
        matched_ids = {r[0] for r in result}
        to_del = []
        for tid in list(self.tracks.keys()):
            if tid not in matched_ids:
                self.tracks[tid]["age"] += 1
                if self.tracks[tid]["age"] > self.max_age:
                    to_del.append(tid)
        for tid in to_del:
            del self.tracks[tid]

        return result


# ─── DEMO ─────────────────────────────────────────────────────────────────────

print("Simple Object Tracker (Centroid Tracking)")
print("=" * 55)
print(f"Generating {N_FRAMES} synthetic frames with 2 moving objects...")

sequence = make_sequence(N_FRAMES)
tracker  = CentroidTracker(max_distance=60)
track_histories = {}   # id -> list of (cx, cy) per frame

TRACK_COLOURS = ["#1565C0", "#B71C1C", "#1B5E20", "#4A148C",
                 "#E65100", "#880E4F"]

all_results = []
for frame_idx, (frame, gt_boxes) in enumerate(sequence):
    dets     = detect_centroids(frame)
    assigned = tracker.update(dets)
    all_results.append((frame, assigned))
    for tid, cx, cy, *_ in assigned:
        if tid not in track_histories:
            track_histories[tid] = []
        track_histories[tid].append((frame_idx, cx, cy))

print(f"  Tracking complete. Found {len(track_histories)} unique track IDs.")

# ─── VISUALISE: 2-row grid (frames 0-4 top, 5-9 bottom) + trajectory ─────────

fig = plt.figure(figsize=(16, 8))
gs  = fig.add_gridspec(3, 5, hspace=0.4, wspace=0.1)

for f_idx, (frame, assigned) in enumerate(all_results):
    row = f_idx // 5
    col = f_idx % 5
    ax  = fig.add_subplot(gs[row, col])
    vis = frame.copy()
    for tid, cx, cy, x1, y1, x2, y2 in assigned:
        c = plt.matplotlib.colors.to_rgb(TRACK_COLOURS[tid % len(TRACK_COLOURS)])
        c_bgr = (int(c[2]*255), int(c[1]*255), int(c[0]*255))
        cv2.rectangle(vis, (x1, y1), (x2, y2), c_bgr, 2)
        cv2.putText(vis, f"ID{tid}", (x1, max(y1-4, 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, c_bgr, 1)
    ax.imshow(vis)
    ax.set_title(f"Frame {f_idx}", fontsize=7)
    ax.axis("off")

# Trajectory plot (row 2, spans all cols)
ax_traj = fig.add_subplot(gs[2, :])
for tid, hist in track_histories.items():
    frames_t = [h[0] for h in hist]
    xs       = [h[1] for h in hist]
    ys       = [h[2] for h in hist]
    col      = TRACK_COLOURS[tid % len(TRACK_COLOURS)]
    ax_traj.plot(xs, [IMG_SIZE - y for y in ys], "o-",
                 color=col, label=f"Track {tid}", linewidth=2, markersize=5)
    # Mark start and end
    ax_traj.scatter([xs[0]], [IMG_SIZE - ys[0]], s=80, color=col, marker="s", zorder=5)
    ax_traj.scatter([xs[-1]], [IMG_SIZE - ys[-1]], s=80, color=col, marker="*", zorder=5)

ax_traj.set_xlim(0, IMG_SIZE); ax_traj.set_ylim(0, IMG_SIZE)
ax_traj.set_title("Object Trajectories (square=start, star=end)", fontsize=9)
ax_traj.legend(fontsize=8, loc="upper right")
ax_traj.grid(alpha=0.3)
ax_traj.set_xlabel("X position"); ax_traj.set_ylabel("Y position (flipped)")

fig.suptitle(
    "Centroid Object Tracker  -- Tracks objects across 10 synthetic frames\n"
    "Detection: colour thresholding + contours | Association: min centroid distance",
    fontsize=10
)
plt.savefig("CNN/outputs/object_tracker.png", dpi=90)
plt.close()
print("Plot saved -> CNN/outputs/object_tracker.png")
print()
print("HOW THIS MAPS TO REAL TRACKERS:")
print("  - This demo: colour contour detector + centroid matching")
print("  - SORT: YOLO CNN detector + Kalman filter + IoU matching")
print("  - DeepSORT: SORT + re-ID embedding network for robust ID assignment")
print("  - ByteTrack: uses low-confidence detections to recover lost tracks")
print("[DONE] simple_object_tracker.py complete")

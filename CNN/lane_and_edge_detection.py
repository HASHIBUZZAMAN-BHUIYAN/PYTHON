"""
lane_and_edge_detection.py
==========================
What it does:
    Generates 3 synthetic road images in NumPy/Matplotlib, then applies a
    classical computer-vision pipeline (Gaussian blur -> Canny edge detection ->
    Hough line transform using OpenCV) to detect lane lines. Detected lines are
    drawn in red over each original image and saved as a comparison grid.

What it teaches:
    - Synthetic image creation for CV testing
    - Classical edge detection (Gaussian blur, Canny)
    - Probabilistic Hough Line Transform for lane detection
    - Difference between rule-based CV and learned CNN approaches

RAM estimate  : ~100 MB peak
Time estimate : < 10 seconds on CPU
Real vs simulated: SIMULATED. Images are generated procedurally with NumPy.
    No real camera feed or real road footage is used.

HOW THIS DIFFERS FROM A CNN APPROACH:
    - Classical CV (used here): hand-engineered filters (Canny, Hough).
      Sensitive to lighting, image quality, and parameter tuning.
      Fast, interpretable, but brittle.
    - CNN approach: learns feature detectors from labelled data.
      More robust to variation in lighting/perspective, but needs training data
      and is a "black box" compared to the explicit Hough maths.
"""

import os
import numpy as np
import cv2

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUTPUT_DIR = "CNN/outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

IMG_H = 128
IMG_W = 128


# -------------------------------------------------------------------------
# Synthetic road image generators
# -------------------------------------------------------------------------
def make_road_image_straight():
    """Straight road with two white lane lines."""
    img = np.zeros((IMG_H, IMG_W, 3), dtype=np.uint8)
    # Sky
    img[:60, :] = [100, 130, 160]
    # Road (gray)
    img[60:, :] = [70, 70, 70]
    # Left lane line (white, ~1/3 from left)
    for y in range(60, IMG_H, 10):
        img[y:y+5, 35:38] = [255, 255, 255]
    # Right lane line (white, ~2/3 from left)
    for y in range(60, IMG_H, 10):
        img[y:y+5, 90:93] = [255, 255, 255]
    return img


def make_road_image_diagonal():
    """Road with angled/perspective lane lines simulating a curve."""
    img = np.zeros((IMG_H, IMG_W, 3), dtype=np.uint8)
    img[:55, :] = [110, 140, 170]
    img[55:, :] = [65, 65, 65]
    # Left lane line - angled
    pt1_l = (30, IMG_H)
    pt2_l = (55, 55)
    cv2.line(img, pt1_l, pt2_l, (255, 255, 255), 2)
    # Right lane line - angled
    pt1_r = (100, IMG_H)
    pt2_r = (73, 55)
    cv2.line(img, pt1_r, pt2_r, (255, 255, 255), 2)
    return img


def make_road_image_noisy():
    """Straight road with noise and a yellow center line."""
    rng = np.random.default_rng(7)
    img = np.zeros((IMG_H, IMG_W, 3), dtype=np.uint8)
    img[:60, :] = [90, 120, 150]
    img[60:, :] = [75, 75, 75]
    # Add road texture noise
    noise = rng.integers(-15, 15, (IMG_H - 60, IMG_W, 3))
    img[60:] = np.clip(img[60:].astype(np.int16) + noise, 0, 255).astype(np.uint8)
    # Yellow center dashed line
    for y in range(60, IMG_H, 8):
        img[y:y+4, 62:65] = [210, 200, 0]
    # White edge lines
    img[60:, 20:23] = [220, 220, 220]
    img[60:, 105:108] = [220, 220, 220]
    return img


# -------------------------------------------------------------------------
# Lane detection pipeline
# -------------------------------------------------------------------------
def detect_lanes(img_bgr):
    """
    Apply Gaussian blur -> Canny edge detection -> Hough line transform.
    Returns the annotated image and the number of lines detected.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # Gaussian blur to reduce noise before edge detection
    blurred = cv2.GaussianBlur(gray, (5, 5), 1.5)

    # Canny edge detection - finds rapid intensity changes (edges)
    edges = cv2.Canny(blurred, threshold1=50, threshold2=150)

    # Probabilistic Hough Line Transform - finds line segments in edge map
    # rho=1 (pixel resolution), theta=pi/180 (angle resolution), threshold=40
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=30,
        minLineLength=20,
        maxLineGap=10,
    )

    # Draw detected lines in red on a copy of the original
    annotated = img_bgr.copy()
    n_lines = 0
    if lines is not None:
        n_lines = len(lines)
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(annotated, (x1, y1), (x2, y2), (0, 0, 255), 2)  # BGR red

    return annotated, edges, n_lines


# -------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("Lane and Edge Detection - Classical CV Pipeline")
    print("=" * 60)
    print("Generating 3 synthetic road images ...")

    images = [
        ("Straight road", make_road_image_straight()),
        ("Diagonal/perspective road", make_road_image_diagonal()),
        ("Noisy road with center line", make_road_image_noisy()),
    ]

    results = []
    for name, img in images:
        annotated, edges, n_lines = detect_lanes(img)
        print(f"  [{name}]: {n_lines} lines detected")
        results.append((name, img, edges, annotated, n_lines))

    # Save a 3-row x 3-col comparison grid
    # Columns: original | edge map | annotated
    fig, axes = plt.subplots(3, 3, figsize=(10, 9))
    col_titles = ["Original", "Canny Edges", "Detected Lines (red)"]
    for col, title in enumerate(col_titles):
        axes[0, col].set_title(title, fontsize=10, fontweight="bold")

    for row, (name, orig, edges, annotated, n_lines) in enumerate(results):
        # OpenCV images are BGR; convert to RGB for matplotlib
        orig_rgb = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)
        annot_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

        axes[row, 0].imshow(orig_rgb)
        axes[row, 0].set_ylabel(name, fontsize=8)
        axes[row, 1].imshow(edges, cmap="gray")
        axes[row, 2].imshow(annot_rgb)
        for col in range(3):
            axes[row, col].axis("off")
        axes[row, 2].set_title(f"{n_lines} lines", fontsize=8, color="red")

    plt.suptitle("Lane Detection: Classical CV (Gaussian -> Canny -> Hough)", fontsize=11)
    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, "lane_detection.png")
    plt.savefig(output_path, dpi=80)
    plt.close()
    print(f"\nComparison grid saved -> {output_path}")
    print("\n[OK] lane_and_edge_detection.py completed successfully.")


if __name__ == "__main__":
    main()

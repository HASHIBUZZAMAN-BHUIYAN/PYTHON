"""
human_pose_keypoints.py
=======================
What it does:
    Generates 5 synthetic stick-figure images using OpenCV drawing primitives,
    then detects joint (keypoint) locations using classical computer vision:
    grayscale thresholding -> binary mask -> cv2.findContours -> centroid
    calculation. Visualizes original figures with detected keypoints overlaid.

What it teaches:
    - Procedural stick-figure image synthesis
    - Binary thresholding for object segmentation
    - Connected-component analysis via cv2.findContours
    - Centroid calculation from image moments
    - Gap between simplified classical CV and real pose estimation

RAM estimate  : ~100 MB peak
Time estimate : < 5 seconds on CPU
Real vs simulated: SIMULATED. All figures are drawn programmatically; no real
    human images or pretrained pose estimation models are used.

NOTE: This is a simplified classical-CV simulation, not a real pose
    estimation model. Real systems (OpenPose, MediaPipe, etc.) use deep
    convolutional networks trained on thousands of labelled human images
    to handle occlusion, perspective, clothing, and body diversity.
"""

import os
import numpy as np
import cv2

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUTPUT_DIR = "CNN/outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

IMG_SIZE = 128


# -------------------------------------------------------------------------
# Stick-figure drawing helpers
# -------------------------------------------------------------------------
def draw_stick_figure(img, cx, cy, scale=1.0, angle_offset=0.0):
    """
    Draw a stick figure centered at (cx, cy) with given scale.
    Joints (circles): head, neck, shoulders (L/R), elbows (L/R), wrists (L/R),
                      hips (L/R), knees (L/R), ankles (L/R)  -> 13 joints total
    Lines connect them as a skeleton.

    Parameters
    ----------
    img          : uint8 BGR image (modified in-place)
    cx, cy       : center-of-hips x, y
    scale        : size multiplier (default 1.0)
    angle_offset : slight arm/leg angle variation (radians)
    """
    s = scale

    # Joint positions (relative to hip center, +y = down)
    joints = {
        "head":       (cx,          cy - int(60 * s)),
        "neck":       (cx,          cy - int(45 * s)),
        "l_shoulder": (cx - int(18 * s), cy - int(40 * s)),
        "r_shoulder": (cx + int(18 * s), cy - int(40 * s)),
        "l_elbow":    (cx - int(28 * s), cy - int(20 * s)),
        "r_elbow":    (cx + int(28 * s), cy - int(20 * s)),
        "l_wrist":    (cx - int(32 * s), cy),
        "r_wrist":    (cx + int(32 * s), cy),
        "l_hip":      (cx - int(12 * s), cy),
        "r_hip":      (cx + int(12 * s), cy),
        "l_knee":     (cx - int(14 * s), cy + int(28 * s)),
        "r_knee":     (cx + int(14 * s), cy + int(28 * s)),
        "l_ankle":    (cx - int(14 * s), cy + int(52 * s)),
        "r_ankle":    (cx + int(14 * s), cy + int(52 * s)),
    }

    # Skeleton connectivity
    limbs = [
        ("head",       "neck"),
        ("neck",       "l_shoulder"),
        ("neck",       "r_shoulder"),
        ("l_shoulder", "l_elbow"),
        ("r_shoulder", "r_elbow"),
        ("l_elbow",    "l_wrist"),
        ("r_elbow",    "r_wrist"),
        ("neck",       "l_hip"),
        ("neck",       "r_hip"),
        ("l_hip",      "l_knee"),
        ("r_hip",      "r_knee"),
        ("l_knee",     "l_ankle"),
        ("r_knee",     "r_ankle"),
    ]

    WHITE = (255, 255, 255)
    JOINT_COLOR = (255, 255, 255)

    # Draw limb lines first (so joint circles appear on top)
    for a, b in limbs:
        pt_a = joints[a]
        pt_b = joints[b]
        cv2.line(img, pt_a, pt_b, WHITE, 2)

    for name, pt in joints.items():
        r = 5 if name == "head" else 3
        cv2.circle(img, pt, r, JOINT_COLOR, -1)

    return joints


def make_figure_image(pose_index):
    """
    Generate one 128x128 black background image with a white stick figure.
    pose_index controls slight positional variation.
    """
    img = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
    rng = np.random.default_rng(pose_index * 17 + 3)
    cx = int(IMG_SIZE // 2 + rng.integers(-8, 8))
    cy = int(IMG_SIZE // 2 + rng.integers(-5, 5))
    scale = 0.7 + rng.integers(0, 3) * 0.1
    joints = draw_stick_figure(img, cx, cy, scale=scale)
    return img, joints


# -------------------------------------------------------------------------
# Keypoint detection via thresholding + findContours
# -------------------------------------------------------------------------
def detect_keypoints(img_bgr):
    """
    Detect bright circular blobs (joint circles) using:
      1. Convert to grayscale
      2. Binary threshold (bright pixels only)
      3. Morphological opening to separate touching blobs
      4. findContours -> compute centroid of each contour
    Returns list of (x, y) centroid tuples.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # Threshold: keep pixels above 200 (the white joints and limb lines)
    _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

    # Erode to break connections between joints and limb lines
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    eroded = cv2.erode(binary, kernel, iterations=2)

    # Dilate slightly to restore joint blobs after erosion
    dilated = cv2.dilate(eroded, kernel, iterations=1)

    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    keypoints = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 2:   # ignore tiny noise specks
            continue
        M = cv2.moments(cnt)
        if M["m00"] == 0:
            continue
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        keypoints.append((cx, cy))

    return keypoints


# -------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("Human Pose Keypoint Detection - Classical CV Simulation")
    print("=" * 60)
    print("NOTE: Simplified classical-CV simulation, not a real pose model.")
    print()

    # Expected number of visible joints per figure
    EXPECTED_JOINTS = 14  # head + neck + 2 shoulders + 2 elbows + 2 wrists
                          # + 2 hips + 2 knees + 2 ankles

    n_figures = 5
    figure_data = []
    for i in range(n_figures):
        img, gt_joints = make_figure_image(i)
        kpts = detect_keypoints(img)
        n_det = len(kpts)
        status = "[OK]" if n_det >= EXPECTED_JOINTS - 3 else "[PARTIAL]"
        print(f"  Figure {i+1}: expected ~{EXPECTED_JOINTS} keypoints, "
              f"detected {n_det}  {status}")
        figure_data.append((img, gt_joints, kpts))

    fig, axes = plt.subplots(n_figures, 2, figsize=(6, 14))
    fig.suptitle("Stick Figure Pose Keypoint Detection (Classical CV)", fontsize=11)
    axes[0, 0].set_title("Generated Figure", fontsize=9)
    axes[0, 1].set_title("Detected Keypoints (blue dots)", fontsize=9)

    for row, (img, gt_joints, kpts) in enumerate(figure_data):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        overlay = img_rgb.copy()
        for (kx, ky) in kpts:
            cv2.circle(overlay, (kx, ky), 4, (0, 100, 255), -1)

        axes[row, 0].imshow(img_rgb)
        axes[row, 0].set_ylabel(f"Figure {row+1}", fontsize=8)
        axes[row, 0].axis("off")

        axes[row, 1].imshow(overlay)
        axes[row, 1].set_title(f"{len(kpts)} pts", fontsize=7)
        axes[row, 1].axis("off")

    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, "pose_keypoints.png")
    plt.savefig(output_path, dpi=80)
    plt.close()
    print(f"\nVisualization saved -> {output_path}")
    print("\n[OK] human_pose_keypoints.py completed successfully.")


if __name__ == "__main__":
    main()

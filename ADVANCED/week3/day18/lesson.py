# Advanced Day 18 — Computer Vision for Robotics (OpenCV)
# ~60 MB RAM, <5s on CPU
# NOTE: heavier lesson — close other apps before running

import cv2
import numpy as np
import matplotlib.pyplot as plt

print("""
=== Computer Vision for Robotics ===

We use synthetic images (generated in code) — no camera required.

Topics:
  1. Color spaces and thresholding
  2. Edge detection (Canny, Sobel)
  3. Contour detection & shape recognition
  4. Color blob tracking (HSV)
  5. Simulated optical flow (Lucas-Kanade)
""")

# ─── Helper: show multiple images ────────────────────────────────────────────
def show_row(images, titles, filename, cmap_list=None):
    n = len(images)
    fig, axes = plt.subplots(1, n, figsize=(4*n, 4))
    if n == 1: axes = [axes]
    for i, (img, title) in enumerate(zip(images, titles)):
        cm = cmap_list[i] if cmap_list else ("gray" if img.ndim==2 else None)
        axes[i].imshow(img, cmap=cm)
        axes[i].set_title(title, fontsize=9); axes[i].axis("off")
    plt.tight_layout(); plt.savefig(filename, dpi=85); plt.close()
    print(f"  Saved {filename}")

# ─── 1. SYNTHETIC SCENE ──────────────────────────────────────────────────────
print("=== 1. Creating synthetic scene ===")
H, W = 300, 400
scene = np.zeros((H, W, 3), dtype=np.uint8)

# Background gradient
for row in range(H):
    v = int(30 + row * 0.1)
    scene[row, :] = (v, v, v+20)

# Colored shapes
cv2.circle(scene,    (80,  80),  50, (0,200,200), -1)    # cyan circle
cv2.rectangle(scene, (180, 60), (280, 160), (200,100,0), -1)   # orange rect
cv2.ellipse(scene,   (330,100),  (50,35), 20, 0, 360, (100,200,100), -1)  # green ellipse
cv2.circle(scene,    (100, 220), 40, (0,0,220), -1)            # red circle
cv2.rectangle(scene, (200,190), (380,270), (200,0,200), -1)    # magenta rect
# Add noise
noise = np.random.randint(0, 20, scene.shape, dtype=np.uint8)
scene = cv2.add(scene, noise)

cv2.imwrite("scene.png", scene)
print("  Saved scene.png")

# ─── 2. COLOR SPACES & THRESHOLDING ─────────────────────────────────────────
print("\n=== 2. Color Spaces & Thresholding ===")
gray  = cv2.cvtColor(scene, cv2.COLOR_BGR2GRAY)
hsv   = cv2.cvtColor(scene, cv2.COLOR_BGR2HSV)

# Otsu's thresholding on grayscale
_, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Adaptive thresholding (handles uneven lighting)
adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                  cv2.THRESH_BINARY, 21, 4)

scene_rgb = cv2.cvtColor(scene, cv2.COLOR_BGR2RGB)
show_row([scene_rgb, gray, otsu, adaptive],
         ["Scene (RGB)", "Grayscale", "Otsu threshold", "Adaptive threshold"],
         "cv_thresholding.png")

# ─── 3. EDGE DETECTION ───────────────────────────────────────────────────────
print("\n=== 3. Edge Detection ===")
blurred = cv2.GaussianBlur(gray, (5,5), 0)

canny = cv2.Canny(blurred, threshold1=50, threshold2=150)

sobelx = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
sobely = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
sobel_mag = np.clip(np.sqrt(sobelx**2 + sobely**2), 0, 255).astype(np.uint8)

laplacian = cv2.Laplacian(blurred, cv2.CV_64F)
laplacian_abs = np.clip(np.abs(laplacian), 0, 255).astype(np.uint8)

show_row([canny, sobel_mag, laplacian_abs],
         ["Canny", "Sobel magnitude", "Laplacian"],
         "cv_edges.png", cmap_list=["gray","gray","gray"])

# ─── 4. CONTOUR DETECTION ───────────────────────────────────────────────────
print("\n=== 4. Contours & Shape Recognition ===")

def classify_shape(contour):
    peri = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.04*peri, True)
    n = len(approx)
    if n == 3: return "triangle"
    if n == 4:
        x,y,w,h = cv2.boundingRect(approx)
        aspect = w/float(h)
        return "square" if 0.85<=aspect<=1.15 else "rectangle"
    return "circle" if n>8 else f"poly({n})"

contour_scene = cv2.cvtColor(scene.copy(), cv2.COLOR_BGR2RGB)
contours, _ = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
for cnt in contours:
    if cv2.contourArea(cnt) < 200: continue
    shape = classify_shape(cnt)
    M = cv2.moments(cnt)
    if M["m00"] == 0: continue
    cx = int(M["m10"]/M["m00"]); cy = int(M["m01"]/M["m00"])
    cv2.drawContours(contour_scene, [cnt], -1, (255,255,0), 2)
    cv2.putText(contour_scene, shape, (cx-20, cy), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (255,255,255), 1)
print(f"  Found {len([c for c in contours if cv2.contourArea(c)>200])} significant contours")
show_row([scene_rgb, contour_scene],
         ["Original", "Detected shapes"],
         "cv_contours.png")

# ─── 5. COLOR BLOB TRACKING ─────────────────────────────────────────────────
print("\n=== 5. Color Blob Tracking (HSV) ===")

# Detect the cyan circle: Hue ~90 in BGR→HSV
lower_cyan = np.array([80, 100, 100])
upper_cyan = np.array([100, 255, 255])
mask_cyan  = cv2.inRange(hsv, lower_cyan, upper_cyan)

# Detect red circle: Hue ~0/180
lower_red1 = np.array([0,  100, 100]); upper_red1 = np.array([10, 255,255])
lower_red2 = np.array([170,100, 100]); upper_red2 = np.array([180,255,255])
mask_red = cv2.bitwise_or(cv2.inRange(hsv, lower_red1, upper_red1),
                           cv2.inRange(hsv, lower_red2, upper_red2))

# Clean masks
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))
mask_cyan = cv2.morphologyEx(mask_cyan, cv2.MORPH_CLOSE, kernel)
mask_red  = cv2.morphologyEx(mask_red,  cv2.MORPH_CLOSE, kernel)

track_img = scene_rgb.copy()
for mask, color, name in [(mask_cyan,(0,220,220),"cyan"), (mask_red,(220,0,0),"red")]:
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in cnts:
        if cv2.contourArea(cnt) < 500: continue
        (x,y),radius = cv2.minEnclosingCircle(cnt)
        cx,cy = int(x),int(y); r=int(radius)
        cv2.circle(track_img,(cx,cy),r,(255,255,255),2)
        cv2.putText(track_img,name,(cx-15,cy-r-5),cv2.FONT_HERSHEY_SIMPLEX,0.5,color,2)

show_row([scene_rgb, cv2.cvtColor(mask_cyan, cv2.COLOR_GRAY2RGB),
          cv2.cvtColor(mask_red, cv2.COLOR_GRAY2RGB), track_img],
         ["Scene","Cyan mask","Red mask","Tracked blobs"],
         "cv_tracking.png")

# ─── 6. OPTICAL FLOW (SIMULATED) ─────────────────────────────────────────────
print("\n=== 6. Lucas-Kanade Optical Flow ===")
frame1 = gray.copy()
# Simulate motion: shift the scene by (5, 3) pixels → "next frame"
M_shift = np.float32([[1,0,5],[0,1,3]])
frame2 = cv2.warpAffine(frame1, M_shift, (W,H))

# Shi-Tomasi corner detector for feature points
corners = cv2.goodFeaturesToTrack(frame1, maxCorners=60, qualityLevel=0.01,
                                   minDistance=15, blockSize=5)
if corners is not None:
    pts1 = corners.astype(np.float32)
    lk_params = dict(winSize=(15,15), maxLevel=2,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    pts2, status, _ = cv2.calcOpticalFlowPyrLK(frame1, frame2, pts1, None, **lk_params)

    flow_img = cv2.cvtColor(frame1, cv2.COLOR_GRAY2RGB)
    for (p1,p2,s) in zip(pts1, pts2, status.ravel()):
        if s:
            x0,y0 = p1.ravel().astype(int)
            x1,y1 = p2.ravel().astype(int)
            cv2.arrowedLine(flow_img,(x0,y0),(x1,y1),(0,220,0),1,tipLength=0.4)
            cv2.circle(flow_img,(x0,y0),3,(255,100,0),-1)

    show_row([cv2.cvtColor(frame1,cv2.COLOR_GRAY2RGB),
              cv2.cvtColor(frame2,cv2.COLOR_GRAY2RGB), flow_img],
             ["Frame 1","Frame 2 (shifted)","LK Optical Flow"],
             "cv_optflow.png")
    print(f"  Tracked {status.ravel().sum()} / {len(pts1)} feature points")
else:
    print("  No corners found")

print("\nDay 18 complete — check saved PNG files.")

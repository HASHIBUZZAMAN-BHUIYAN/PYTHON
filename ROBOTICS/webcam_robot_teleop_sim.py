"""
Webcam Robot Teleop Simulation (HSV Colour Tracking + Fallback)
================================================================
What it does:
  A simulated 2D ground robot is driven by a "joystick" signal derived
  from tracking a coloured object:

  WEBCAM MODE (if a webcam is detected and working):
    - Opens the camera with OpenCV
    - Applies HSV thresholding to find the largest blob of a target colour
      (default: green; adjustable via HSV_LOWER / HSV_UPPER constants)
    - Maps blob position relative to frame center -> robot velocity command
      v_x = K_TELE * (blob_x - cx) / cx
      v_y = K_TELE * (blob_y - cy) / cy
    - Draws the tracked blob and robot state overlaid on the live video feed
    - Press 'q' or run for N_CAM_FRAMES to stop

  SYNTHETIC FALLBACK MODE (no webcam, or webcam fails to open):
    - Generates a simulated "tracked blob" that follows a Lissajous figure-8
    - Shows a matplotlib animation with the simulated blob (left) and the
      robot arena (right), updating in real time
    - Identical robot dynamics to webcam mode -- only the input source differs

  Robot dynamics (both modes):
    v_cmd = K_TELE * (blob_offset from center) [+ small Gaussian noise]
    pos_new = pos + v_cmd * DT   (clamped to arena bounds)

What it teaches:
  - HSV colour space: why it's easier to threshold colours than RGB/BGR
  - Proportional tele-operation: joystick position -> velocity command
  - Why colour tracking fails (illumination changes, colour confusion)
  - The design pattern: same robot logic, interchangeable input sources
  - Graceful degradation: code always works even without hardware

Controls (webcam mode): Hold a green object in front of the camera.
                        The simulated robot will follow it.  Press 'q' to quit.
Controls (fallback):    None -- auto-runs for N_SYN_STEPS then exits.
Output: ROBOTICS/outputs/webcam_teleop.png
RAM: ~80 MB | Time: ~10s (webcam) or ~5s (fallback)
"""

import os, sys, time, math
import numpy as np
import matplotlib
# Check mode before importing pyplot so backend decision is correct
_HEADLESS = os.environ.get('MPLBACKEND', '').lower() == 'agg'
INTERACTIVE = not _HEADLESS
import matplotlib.pyplot as plt
import matplotlib.patches as patches

os.makedirs("ROBOTICS/outputs", exist_ok=True)

# ─── CONFIGURATION ────────────────────────────────────────────────────────────
# HSV range for target colour (green by default)
HSV_LOWER = np.array([35,  60, 60], dtype=np.uint8)
HSV_UPPER = np.array([85, 255, 255], dtype=np.uint8)
MIN_BLOB_AREA = 500      # min contour area to count as "found"

K_TELE    = 2.5          # joystick-to-velocity gain
DT        = 0.05
MAX_SPD   = 3.0
NOISE     = 0.05         # tiny noise to show robot isn't frozen

# Robot arena (shared between both modes)
W_ARENA, H_ARENA = 10.0, 10.0
robot_pos = np.array([W_ARENA/2, H_ARENA/2], dtype=float)
robot_vel = np.zeros(2)
robot_path = [robot_pos.copy()]

N_CAM_FRAMES = 300       # stop after this many webcam frames
N_SYN_STEPS  = 300       # synthetic fallback frames
DRAW_N = 3

rng = np.random.default_rng(42)

# ─── TRY TO OPEN WEBCAM ───────────────────────────────────────────────────────
WEBCAM = False
cap = None
try:
    import cv2 as cv
    cap = cv.VideoCapture(0)
    ret, test_frame = cap.read()
    if ret and test_frame is not None and test_frame.size > 0:
        WEBCAM = True
        print(f"Webcam opened: {test_frame.shape[1]}x{test_frame.shape[0]}")
    else:
        cap.release()
        cap = None
        print("Webcam not available -- switching to synthetic fallback")
except Exception as e:
    print(f"Webcam unavailable ({e}) -- switching to synthetic fallback")

# ─── SYNTHETIC FIGURE-8 INPUT ─────────────────────────────────────────────────
def synthetic_blob(step):
    """Lissajous figure-8 blob position in normalised (-1..1) coords."""
    t = step * DT
    bx = 0.85 * math.sin(t * 0.6)
    by = 0.85 * math.sin(t * 1.2)
    return np.array([bx, by])   # range [-1, 1] in each axis

# ─── SHARED ROBOT UPDATE ──────────────────────────────────────────────────────
def update_robot(joystick_norm):
    """joystick_norm: normalised offset in [-1,1]^2 (relative to center)."""
    global robot_pos, robot_vel
    v_cmd = K_TELE * joystick_norm + rng.normal(0, NOISE, 2)
    spd = np.linalg.norm(v_cmd)
    if spd > MAX_SPD:
        v_cmd = v_cmd / spd * MAX_SPD
    robot_vel = v_cmd
    robot_pos = np.clip(robot_pos + robot_vel * DT, 0.0, [W_ARENA, H_ARENA])
    robot_path.append(robot_pos.copy())

# ══════════════════════════════════════════════════════════════════════════════
# MODE A: WEBCAM
# ══════════════════════════════════════════════════════════════════════════════
if WEBCAM:
    print("\nWEBCAM MODE: Hold a GREEN object in front of the camera.")
    print(f"Running for {N_CAM_FRAMES} frames or press 'q' to quit.\n")
    import cv2 as cv

    frame_h, frame_w = test_frame.shape[:2]
    cx, cy = frame_w // 2, frame_h // 2
    blob_xy = (cx, cy)

    # Draw robot in the BGR frame using cv2
    def robot_to_px(pos):
        """Convert robot arena coords to frame pixel coords."""
        px = int(pos[0] / W_ARENA * frame_w)
        py = int((1.0 - pos[1] / H_ARENA) * frame_h)  # flip y
        return px, py

    last_frame = test_frame.copy()
    for frame_idx in range(N_CAM_FRAMES):
        ret, frame = cap.read()
        if not ret:
            break

        # HSV blob detection
        hsv   = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
        mask  = cv.inRange(hsv, HSV_LOWER, HSV_UPPER)
        mask  = cv.morphologyEx(mask, cv.MORPH_OPEN, np.ones((5,5), np.uint8))
        cnts, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        found = False
        if cnts:
            biggest = max(cnts, key=cv.contourArea)
            if cv.contourArea(biggest) > MIN_BLOB_AREA:
                M = cv.moments(biggest)
                if M['m00'] > 0:
                    bx = int(M['m10'] / M['m00'])
                    by = int(M['m01'] / M['m00'])
                    blob_xy = (bx, by)
                    found = True
                    cv.drawContours(frame, [biggest], -1, (0, 255, 0), 2)
                    cv.circle(frame, blob_xy, 8, (0, 255, 0), -1)

        # Normalised joystick
        joy_x = (blob_xy[0] - cx) / cx
        joy_y = (blob_xy[1] - cy) / cy
        update_robot(np.array([joy_x, joy_y]))

        rpx, rpy = robot_to_px(robot_pos)
        cv.circle(frame, (rpx, rpy), 18, (255, 80, 0), -1)
        cv.circle(frame, (rpx, rpy), 18, (255, 255, 255), 2)

        if len(robot_path) > 1:
            for k in range(max(0, len(robot_path)-30), len(robot_path)-1):
                pa = robot_to_px(robot_path[k])
                pb = robot_to_px(robot_path[k+1])
                cv.line(frame, pa, pb, (255, 150, 0), 1)

        status = "TRACKING" if found else "NO BLOB"
        cv.putText(frame, f"Teleop: {status}  frame {frame_idx}/{N_CAM_FRAMES}",
                   (10, 25), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
        cv.putText(frame, f"Joy: ({joy_x:+.2f}, {joy_y:+.2f})",
                   (10, 55), cv.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 1)
        cv.line(frame, (cx-20, cy), (cx+20, cy), (100,100,255), 1)
        cv.line(frame, (cx, cy-20), (cx, cy+20), (100,100,255), 1)

        last_frame = frame.copy()
        if INTERACTIVE:
            cv.imshow("Webcam Teleop", frame)
            if cv.waitKey(1) & 0xFF == ord('q'):
                print("User pressed 'q' -- stopping")
                break

    cap.release()
    if INTERACTIVE:
        cv.destroyAllWindows()

    # Save summary plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    # Show final camera frame (convert BGR->RGB)
    ax1.imshow(cv.cvtColor(last_frame, cv.COLOR_BGR2RGB))
    ax1.set_title("Final Camera Frame (Webcam Mode)", fontsize=9)
    ax1.axis('off')
    pts = np.array(robot_path)
    ax2.plot(pts[:,0], pts[:,1], 'b-', lw=1.5, alpha=0.7)
    ax2.plot(*pts[0],  'gs', ms=10, label='Start')
    ax2.plot(*pts[-1], 'r*', ms=14, label='End')
    ax2.set_xlim(0, W_ARENA);  ax2.set_ylim(0, H_ARENA)
    ax2.set_aspect('equal');   ax2.grid(alpha=0.3)
    ax2.set_title("Robot Path in Arena", fontsize=9)
    ax2.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig("ROBOTICS/outputs/webcam_teleop.png", dpi=90, bbox_inches='tight')
    print(f"Saved: ROBOTICS/outputs/webcam_teleop.png")
    plt.close()

# ══════════════════════════════════════════════════════════════════════════════
# MODE B: SYNTHETIC FALLBACK
# ══════════════════════════════════════════════════════════════════════════════
else:
    print("\nSYNTHETIC FALLBACK MODE: blob follows Lissajous figure-8")
    print(f"Running for {N_SYN_STEPS} steps.\n")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6))
    fig.suptitle("Webcam Teleop (Synthetic Fallback -- No Webcam)", fontsize=11)
    if INTERACTIVE:
        plt.ion()

    def draw(step, blob_norm):
        ax1.clear();  ax2.clear()

        # Left: "simulated camera" view showing blob
        ax1.set_xlim(-1.1, 1.1);  ax1.set_ylim(-1.1, 1.1)
        ax1.set_aspect('equal');   ax1.set_facecolor('#111111')
        ax1.set_title(f"Simulated Camera View  (step {step}/{N_SYN_STEPS})", fontsize=9)
        ax1.axhline(0, color='grey', lw=0.5);  ax1.axvline(0, color='grey', lw=0.5)
        ax1.add_patch(patches.Circle(blob_norm, 0.08,
                                     facecolor='#00ff44', edgecolor='white', lw=2, zorder=5))
        ax1.text(*blob_norm + np.array([0.08, 0.08]), 'blob', color='#00ff44', fontsize=8)
        ax1.set_xlabel(f"Joystick: ({blob_norm[0]:+.2f}, {blob_norm[1]:+.2f})", fontsize=8)
        # Joystick arrow from center
        ax1.annotate('', xy=blob_norm*0.6, xytext=np.zeros(2),
                     arrowprops=dict(arrowstyle='->', color='cyan', lw=2))
        ax1.add_patch(patches.Rectangle((-1,-1), 2, 2, facecolor='none',
                                         edgecolor='grey', lw=1))
        ax1.set_xticks([]);  ax1.set_yticks([])

        # Right: robot arena
        ax2.set_xlim(-0.3, W_ARENA+0.3);  ax2.set_ylim(-0.3, H_ARENA+0.3)
        ax2.set_aspect('equal');           ax2.set_facecolor('#f8f8f8')
        ax2.set_title("Robot Arena (2D top-down view)", fontsize=9)
        pts = np.array(robot_path)
        ax2.plot(pts[:,0], pts[:,1], '-', color='royalblue', alpha=0.4, lw=1.5)
        ax2.add_patch(patches.Circle(robot_pos, 0.3,
                                     facecolor='royalblue', edgecolor='black', lw=1.5, zorder=5))
        spd = np.linalg.norm(robot_vel)
        if spd > 0.05:
            ax2.annotate('', xy=robot_pos+robot_vel/spd*0.5, xytext=robot_pos,
                         arrowprops=dict(arrowstyle='->', color='blue', lw=2))
        ax2.plot(*robot_path[0], 'gs', ms=9, label='Start', zorder=4)
        ax2.legend(fontsize=8);  ax2.grid(alpha=0.2)
        plt.tight_layout()

    for step in range(N_SYN_STEPS):
        blob_norm = synthetic_blob(step)
        update_robot(blob_norm)
        if step % DRAW_N == 0:
            draw(step, blob_norm)
            if INTERACTIVE:
                plt.pause(0.04)

    draw(N_SYN_STEPS-1, synthetic_blob(N_SYN_STEPS-1))
    total_dist = sum(np.linalg.norm(np.array(robot_path[k+1]) - np.array(robot_path[k]))
                     for k in range(len(robot_path)-1))
    print(f"Run complete. Robot travelled {total_dist:.2f} m in {N_SYN_STEPS} steps.")
    plt.savefig("ROBOTICS/outputs/webcam_teleop.png", dpi=90, bbox_inches='tight')
    print("Saved: ROBOTICS/outputs/webcam_teleop.png")
    if INTERACTIVE:
        plt.ioff();  plt.show()
    plt.close()

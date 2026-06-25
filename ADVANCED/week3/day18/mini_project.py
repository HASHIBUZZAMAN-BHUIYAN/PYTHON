# Advanced Day 18 Mini-Project — Color-Guided Robot Navigation
# Robot detects colored waypoint beacons in a synthetic top-down scene
# and navigates to each one in order.
# ~55 MB RAM, <5s on CPU

import cv2
import numpy as np
import matplotlib.pyplot as plt

print("=== Day 18 Mini-Project: Color-Guided Robot Navigation ===\n")

# ─── Synthetic overhead map ───────────────────────────────────────────────────
H, W = 400, 600
arena = np.ones((H, W, 3), dtype=np.uint8) * 220  # light grey floor

# Walls / obstacles
obstacles = [
    (50, 50, 200, 60),     # top-left block
    (350, 80, 420, 250),   # right-side column
    (120, 200, 130, 380),  # vertical wall
    (250, 300, 500, 320),  # horizontal wall
]
for (x1,y1,x2,y2) in obstacles:
    cv2.rectangle(arena, (x1,y1),(x2,y2),(80,80,80),-1)

# Colored waypoint beacons (circles)
# (center_x, center_y, color_BGR, name, hsv_lower, hsv_upper)
beacons = [
    (80,  300, (0,200,200), "Cyan",    np.array([85,100,100]),  np.array([95,255,255])),
    (460, 150, (0,150,0),   "Green",   np.array([50,100,100]),  np.array([70,255,255])),
    (300, 360, (0,0,200),   "Red",     np.array([0,100,100]),   np.array([10,255,255])),
    (530, 350, (200,100,0), "Orange",  np.array([10,150,150]),  np.array([25,255,255])),
]
for (bx,by,color,_,_,_) in beacons:
    cv2.circle(arena,(bx,by),18,color,-1)
    cv2.circle(arena,(bx,by),18,(0,0,0),2)

# Robot start
robot_pos = np.array([50., 350.], dtype=float)
cv2.circle(arena,(int(robot_pos[0]),int(robot_pos[1])),12,(150,50,200),-1)

cv2.imwrite("arena.png", arena)
print("  Saved arena.png")

# ─── Detect beacons via color thresholding ────────────────────────────────────
hsv_arena = cv2.cvtColor(arena, cv2.COLOR_BGR2HSV)

def detect_beacon(hsv_img, lower, upper, name):
    mask = cv2.inRange(hsv_img, lower, upper)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    centers = []
    for cnt in cnts:
        if cv2.contourArea(cnt) < 200: continue
        M = cv2.moments(cnt)
        if M["m00"] == 0: continue
        cx = int(M["m10"]/M["m00"]); cy = int(M["m01"]/M["m00"])
        centers.append((cx, cy))
    return centers

detected = {}
for bx,by,_,name,lo,hi in beacons:
    centers = detect_beacon(hsv_arena, lo, hi, name)
    if centers:
        detected[name] = centers[0]
        print(f"  Detected {name:8s} beacon at {centers[0]}  (truth: {bx},{by})")
    else:
        # Fallback to known position (red detection can be tricky at the edges)
        detected[name] = (bx, by)
        print(f"  {name:8s}: fallback to known position ({bx},{by})")

# ─── Simple PID path to each beacon ─────────────────────────────────────────
SPEED = 4.0
DT    = 0.1

class PID2D:
    def __init__(self, Kp=1.5, Ki=0.05, Kd=0.3, dt=DT, max_force=8.0):
        self.Kp=Kp; self.Ki=Ki; self.Kd=Kd; self.dt=dt; self.max_force=max_force
        self.integral=np.zeros(2); self.prev_error=np.zeros(2)

    def step(self, error):
        self.integral = np.clip(self.integral + error*self.dt, -20., 20.)
        D = (error - self.prev_error) / self.dt
        self.prev_error = error
        out = self.Kp*error + self.Ki*self.integral + self.Kd*D
        norm = np.linalg.norm(out)
        if norm > self.max_force: out = out/norm * self.max_force
        return out

order = ["Cyan","Green","Red","Orange"]
full_path = [robot_pos.copy()]
segment_info = []

pos = robot_pos.copy()
pid = PID2D()
for beacon_name in order:
    goal = np.array(detected[beacon_name], dtype=float)
    pid.integral[:]=0; pid.prev_error[:]=0
    steps=0; segment=[]
    for _ in range(500):
        err = goal - pos
        dist = np.linalg.norm(err)
        if dist < 6.:
            break
        force = pid.step(err)
        pos = pos + force * DT
        # Clamp to arena
        pos[0] = np.clip(pos[0], 0, W-1)
        pos[1] = np.clip(pos[1], 0, H-1)
        segment.append(pos.copy()); full_path.append(pos.copy()); steps+=1
    segment_info.append((beacon_name, steps, dist))
    print(f"  Navigated to {beacon_name:8s} in {steps} steps, final dist={dist:.1f}px")

# ─── Visualize path on arena ─────────────────────────────────────────────────
vis = cv2.cvtColor(arena.copy(), cv2.COLOR_BGR2RGB)
path_arr = np.array(full_path).astype(int)

# Draw path segments with color coding
colors_draw = [(0,200,200),(0,180,0),(200,50,50),(220,120,0)]
prev_pt = path_arr[0]
seg_idx = 0
steps_acc = 0
thresholds = [info[1] for info in segment_info]
for i, pt in enumerate(path_arr[1:], 1):
    col_idx = min(seg_idx, len(colors_draw)-1)
    cv2.line(vis,(prev_pt[0],prev_pt[1]),(pt[0],pt[1]),colors_draw[col_idx],2)
    prev_pt = pt
    if seg_idx < len(thresholds) and (i - sum(thresholds[:seg_idx])) >= thresholds[seg_idx]:
        seg_idx += 1

# Draw start
cv2.circle(vis,(int(robot_pos[0]),int(robot_pos[1])),10,(180,50,220),-1)
cv2.putText(vis,"START",(int(robot_pos[0])+5,int(robot_pos[1])-12),
            cv2.FONT_HERSHEY_SIMPLEX,0.45,(0,0,0),1)

# Draw detected beacons with labels
for bx,by,_,name,_,_ in beacons:
    cv2.circle(vis,(bx,by),18,(255,255,255),2)
    cv2.putText(vis,name,(bx-18,by-22),cv2.FONT_HERSHEY_SIMPLEX,0.45,(0,0,0),1)

# ─── Summary plot ─────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

axes[0].imshow(cv2.cvtColor(arena, cv2.COLOR_BGR2RGB))
axes[0].set_title("Arena (original)"); axes[0].axis("off")

axes[1].imshow(vis)
axes[1].set_title("Robot navigation path"); axes[1].axis("off")

plt.suptitle("Day 18 Mini-Project — Color-Guided Robot Navigation", fontsize=12)
plt.tight_layout(); plt.savefig("robot_navigation.png", dpi=90)
print("\nSaved robot_navigation.png")
plt.show()

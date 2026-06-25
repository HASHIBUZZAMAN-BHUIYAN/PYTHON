# Advanced Day 18 — Solutions
import cv2
import numpy as np
import matplotlib.pyplot as plt

H, W = 300, 400
scene = np.zeros((H, W, 3), dtype=np.uint8)
for row in range(H):
    v = int(30 + row * 0.1)
    scene[row, :] = (v, v, v+20)
cv2.circle(scene,    (80,80),  50, (0,200,200), -1)
cv2.rectangle(scene, (180,60),(280,160),(200,100,0),-1)
cv2.ellipse(scene,   (330,100),(50,35),20,0,360,(100,200,100),-1)
cv2.circle(scene,    (100,220),40,(0,0,220),-1)
cv2.rectangle(scene, (200,190),(380,270),(200,0,200),-1)
noise = np.random.randint(0,20,scene.shape,dtype=np.uint8)
scene = cv2.add(scene, noise)
gray = cv2.cvtColor(scene, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray,(5,5),0)

# Exercise 1 — Prewitt
print("=== Exercise 1: Prewitt ===")
Px = np.float32([[-1,0,1],[-1,0,1],[-1,0,1]])
Py = np.float32([[-1,-1,-1],[0,0,0],[1,1,1]])
prewx = cv2.filter2D(blurred.astype(np.float32), -1, Px)
prewy = cv2.filter2D(blurred.astype(np.float32), -1, Py)
prewitt = np.clip(np.sqrt(prewx**2+prewy**2),0,255).astype(np.uint8)
sobx = cv2.Sobel(blurred,cv2.CV_64F,1,0,ksize=3); soby=cv2.Sobel(blurred,cv2.CV_64F,0,1,ksize=3)
sobel = np.clip(np.sqrt(sobx**2+soby**2),0,255).astype(np.uint8)
canny = cv2.Canny(blurred,50,150)
fig,axes=plt.subplots(1,3,figsize=(12,4))
for ax,img,t in zip(axes,[prewitt,sobel,canny],["Prewitt","Sobel","Canny"]):
    ax.imshow(img,cmap="gray"); ax.set_title(t); ax.axis("off")
plt.tight_layout(); plt.savefig("s1_prewitt.png",dpi=80); plt.close()
print("  Saved s1_prewitt.png")

# Exercise 2 — Template matching
print("\n=== Exercise 2: Template Matching ===")
template = gray[65:105, 185:225].copy()  # crop from orange rect area
result = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)
_, max_val, _, max_loc = cv2.minMaxLoc(result)
th,tw = template.shape
x1,y1 = max_loc; x2,y2 = x1+tw, y1+th
match_img = cv2.cvtColor(scene.copy(),cv2.COLOR_BGR2RGB)
cv2.rectangle(match_img,(x1,y1),(x2,y2),(255,0,0),3)
print(f"  Best match at {max_loc}, score={max_val:.3f}")
plt.figure(figsize=(6,4)); plt.imshow(match_img); plt.title("Template match"); plt.axis("off")
plt.tight_layout(); plt.savefig("s2_template.png",dpi=80); plt.close()

# Exercise 3 — Perspective transform
print("\n=== Exercise 3: Perspective Transform ===")
src_pts = np.float32([[50,50],[150,40],[160,160],[45,170]])
dst_pts = np.float32([[0,0],[200,0],[200,200],[0,200]])
M_h = cv2.getPerspectiveTransform(src_pts, dst_pts)
warped = cv2.warpPerspective(cv2.cvtColor(scene,cv2.COLOR_BGR2RGB), M_h, (200,200))
fig,axes=plt.subplots(1,2,figsize=(8,4))
orig_show=cv2.cvtColor(scene.copy(),cv2.COLOR_BGR2RGB)
cv2.polylines(orig_show,[src_pts.astype(int)],True,(255,0,0),2)
axes[0].imshow(orig_show); axes[0].set_title("Source quad"); axes[0].axis("off")
axes[1].imshow(warped); axes[1].set_title("Warped (bird's eye)"); axes[1].axis("off")
plt.tight_layout(); plt.savefig("s3_perspective.png",dpi=80); plt.close()
print("  Saved s3_perspective.png")

# Exercise 4 — Distance estimation
print("\n=== Exercise 4: Monocular Distance ===")
REAL_WIDTH = 0.3  # meters
F_PIXELS   = 600.
# Detect orange rectangle via HSV
hsv = cv2.cvtColor(scene, cv2.COLOR_BGR2HSV)
mask_orange = cv2.inRange(hsv, np.array([10,100,100]), np.array([25,255,255]))
kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(9,9))
mask_orange = cv2.morphologyEx(mask_orange,cv2.MORPH_CLOSE,kernel)
cnts,_ = cv2.findContours(mask_orange,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
if cnts:
    cnt = max(cnts,key=cv2.contourArea)
    x,y,w,h = cv2.boundingRect(cnt)
    dist = (REAL_WIDTH * F_PIXELS) / w
    print(f"  Pixel width={w}px → estimated distance = {dist:.3f} m")
else:
    print("  Orange rectangle not detected")

# Exercise 5 — Occupancy grid
print("\n=== Exercise 5: Occupancy Grid ===")
FLOOR_SMALL = np.array([
    [1,1,1,1,1,1,1,1,1,1],
    [1,0,0,0,0,0,1,0,0,1],
    [1,0,0,0,0,0,0,0,0,1],
    [1,0,0,1,1,0,1,0,0,1],
    [1,0,0,0,1,0,0,0,0,1],
    [1,0,0,0,0,0,0,0,0,1],
    [1,1,0,0,1,1,1,0,0,1],
    [1,0,0,0,0,0,0,0,0,1],
    [1,0,0,0,0,0,0,0,0,1],
    [1,1,1,1,1,1,1,1,1,1],
], dtype=int)
R,C=FLOOR_SMALL.shape
occ = np.full((R,C),-1,dtype=int)  # -1=unknown, 0=free, 1=occupied

# Robot follows a simple path
robot_path=[(1,1),(1,2),(1,3),(1,4),(2,4),(3,4),(4,4),(5,4),(5,5),(5,6),(5,7),(6,7),(7,7),(8,7),(8,6),(8,5)]
SCAN_RADIUS=2
for (rr,cc) in robot_path:
    occ[rr,cc]=0
    for dr in range(-SCAN_RADIUS,SCAN_RADIUS+1):
        for dc in range(-SCAN_RADIUS,SCAN_RADIUS+1):
            nr,nc=rr+dr,cc+dc
            if 0<=nr<R and 0<=nc<C:
                if FLOOR_SMALL[nr,nc]==1: occ[nr,nc]=1
                elif occ[nr,nc]==-1: occ[nr,nc]=0

cmap_occ = np.ones((R,C,3))*0.5   # grey = unknown
cmap_occ[occ==0] = [0.9,0.9,0.9]  # white = free
cmap_occ[occ==1] = [0.1,0.1,0.1]  # black = occupied
for (rr,cc) in robot_path: cmap_occ[rr,cc]=[0.,0.7,0.3]
fig,axes=plt.subplots(1,2,figsize=(9,4))
axes[0].imshow(1-FLOOR_SMALL,cmap="gray"); axes[0].set_title("True map"); axes[0].axis("off")
axes[1].imshow(cmap_occ); axes[1].set_title("Discovered occupancy"); axes[1].axis("off")
plt.tight_layout(); plt.savefig("s5_occupancy.png",dpi=80); plt.close()
print("  Saved s5_occupancy.png")

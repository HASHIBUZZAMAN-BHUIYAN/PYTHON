# Advanced Day 15 — Solutions
import numpy as np, matplotlib.pyplot as plt

L1,L2=1.0,0.8

# Exercise 1
def Rz(a): return np.array([[np.cos(a),-np.sin(a),0],[np.sin(a),np.cos(a),0],[0,0,1]])
def Ry(a): return np.array([[np.cos(a),0,np.sin(a)],[0,1,0],[-np.sin(a),0,np.cos(a)]])
def Rx(a): return np.array([[1,0,0],[0,np.cos(a),-np.sin(a)],[0,np.sin(a),np.cos(a)]])
v=np.array([1.,0.,0.])
print("Rz(90°) @ [1,0,0]:", (Rz(np.pi/2)@v).round(3))
print("Rz(45°)@Ry(30°)@[1,0,0]:", (Rz(np.pi/4)@Ry(np.pi/6)@v).round(3))

# Exercise 2
def fk3(t1,t2,t3,L1=1,L2=0.8,L3=0.5):
    o=np.array([0.,0.])
    j1=o+np.array([L1*np.cos(t1),L1*np.sin(t1)])
    j2=j1+np.array([L2*np.cos(t1+t2),L2*np.sin(t1+t2)])
    ee=j2+np.array([L3*np.cos(t1+t2+t3),L3*np.sin(t1+t2+t3)])
    return o,j1,j2,ee
deg=np.pi/180; o,j1,j2,ee=fk3(30*deg,45*deg,-30*deg)
print("3-link EE:",ee.round(3))
fig,ax=plt.subplots(figsize=(5,5))
pts=[o,j1,j2,ee]
ax.plot([p[0] for p in pts],[p[1] for p in pts],"o-",linewidth=3)
ax.set_aspect("equal"); ax.grid(True); ax.set_title("3-link arm")
plt.savefig("3link.png",dpi=72); plt.close()

# Exercise 3
t1,t2=np.pi/4,np.pi/6
J=np.array([[-L1*np.sin(t1)-L2*np.sin(t1+t2), -L2*np.sin(t1+t2)],
             [L1*np.cos(t1)+L2*np.cos(t1+t2),  L2*np.cos(t1+t2)]])
dtheta=np.array([0.1,0.2])
v_ee=J@dtheta; print("End-effector velocity:",v_ee.round(4))

# Exercise 4
x,y,theta=0.,0.,0.; d=0.3; dt=0.05; path_x,path_y=[x],[y]
for t in np.arange(0,5,dt):
    vl,vr=(0.5,1.0) if 2<=t<3 else (1.0,1.0)
    v=(vr+vl)/2; w=(vr-vl)/d
    x+=v*np.cos(theta)*dt; y+=v*np.sin(theta)*dt; theta+=w*dt
    path_x.append(x); path_y.append(y)
fig,ax=plt.subplots(figsize=(6,5))
ax.plot(path_x,path_y); ax.plot(path_x[0],path_y[0],"g^",ms=10,label="Start"); ax.plot(path_x[-1],path_y[-1],"r*",ms=12,label="End")
ax.set_aspect("equal"); ax.legend(); ax.set_title("Differential-drive path"); plt.savefig("diffdrive.png",dpi=72); plt.close()

# Exercise 5
wx,wy=[],[]
for t1 in np.linspace(0,2*np.pi,60):
    for t2 in np.linspace(-np.pi,np.pi,60):
        ex=L1*np.cos(t1)+L2*np.cos(t1+t2); ey=L1*np.sin(t1)+L2*np.sin(t1+t2)
        wx.append(ex); wy.append(ey)
wx,wy=np.array(wx),np.array(wy)
obs_c=np.array([0.8,0.5]); inside=(wx-obs_c[0])**2+(wy-obs_c[1])**2<0.04
fig,ax=plt.subplots(figsize=(6,6))
ax.scatter(wx[~inside],wy[~inside],s=1,c="green",alpha=0.2)
ax.scatter(wx[inside],wy[inside],s=1,c="red",alpha=0.5)
circle=plt.Circle(obs_c,0.2,color="red",fill=False,linewidth=2); ax.add_patch(circle)
ax.set_aspect("equal"); ax.set_title("Workspace with obstacle"); plt.savefig("workspace_obs.png",dpi=72); plt.close()
print("Saved 3link.png, diffdrive.png, workspace_obs.png")

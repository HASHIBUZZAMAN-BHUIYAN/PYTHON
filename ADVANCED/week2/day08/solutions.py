# Advanced Day 08 — Solutions
import numpy as np, matplotlib.pyplot as plt
from sklearn.datasets import make_circles
np.random.seed(42)

def sigmoid(z): return 1/(1+np.exp(-np.clip(z,-500,500)))

# Exercise 1
x = np.linspace(-5,5,200)
fig,axes = plt.subplots(2,3,figsize=(12,7))
for col,(fn,dfn,name) in enumerate([(sigmoid, lambda z:sigmoid(z)*(1-sigmoid(z)),"sigmoid"),
                                     (np.tanh, lambda z:1-np.tanh(z)**2,"tanh"),
                                     (lambda z:np.maximum(0,z), lambda z:(z>0).astype(float),"relu")]):
    axes[0,col].plot(x,fn(x)); axes[0,col].set_title(name); axes[0,col].grid(True)
    axes[1,col].plot(x,dfn(x),color="orange"); axes[1,col].set_title(f"d({name})/dx"); axes[1,col].grid(True)
plt.tight_layout(); plt.savefig("activations.png",dpi=72); plt.close()

# Exercise 2
X_and = np.array([[0,0],[0,1],[1,0],[1,1]],dtype=float)
y_and = np.array([0,0,0,1],dtype=float)
w = np.random.randn(2)*0.1; b = 0.0; lr = 0.5
for _ in range(1000):
    z = X_and@w + b; a = sigmoid(z)
    dz = a - y_and; w -= lr*(X_and.T@dz)/4; b -= lr*dz.mean()
print("AND perceptron:", (sigmoid(X_and@w+b)>0.5).astype(int), "weights:", w.round(3), "bias:", round(b,3))

# Exercise 3
def bce(y,a): eps=1e-8; return -np.mean(y*np.log(a+eps)+(1-y)*np.log(1-a+eps))
X_c,y_c = make_circles(200,noise=0.1,random_state=42)
W1=np.random.randn(2,4)*0.1; b1=np.zeros((1,4)); W2=np.random.randn(4,1)*0.1; b2=np.zeros((1,1))
eps_grad=1e-5
def forward(X):
    A1=sigmoid(X@W1+b1); return sigmoid(A1@W2+b2), A1
A2_orig,A1_orig=forward(X_c[:4])
loss_orig=bce(y_c[:4],A2_orig.flatten())
W1_perturb=W1.copy(); W1_perturb[0,0]+=eps_grad
A2_p,_=forward(X_c[:4]); loss_p=bce(y_c[:4],A2_p.flatten())
W1_perturb[0,0]-=2*eps_grad
A2_m,_=forward(X_c[:4]); loss_m=bce(y_c[:4],A2_m.flatten())
num_grad=(loss_p-loss_m)/(2*eps_grad)
print(f"Numerical gradient W1[0,0]: {num_grad:.6f}")
print("(Compare to analytical gradient from backprop — should be very close)")

# Exercise 5
x_reg=np.linspace(-3,3,200).reshape(-1,1)
y_reg=(0.5*x_reg.flatten()**2+np.random.normal(0,0.1,200)).reshape(-1,1)
W1r=np.random.randn(1,8)*0.3; b1r=np.zeros((1,8)); W2r=np.random.randn(8,1)*0.1; b2r=np.zeros((1,1))
for _ in range(2000):
    Z1=x_reg@W1r+b1r; A1=np.maximum(0,Z1); Z2=A1@W2r+b2r
    dZ2=(Z2-y_reg)/200; dW2=A1.T@dZ2; dA1=dZ2@W2r.T; dZ1=dA1*(Z1>0)
    dW1=x_reg.T@dZ1; W2r-=0.01*dW2; b2r-=0.01*dZ2.mean(axis=0,keepdims=True)
    W1r-=0.01*dW1; b1r-=0.01*dZ1.mean(axis=0,keepdims=True)
A1f=np.maximum(0,x_reg@W1r+b1r); yp=A1f@W2r+b2r
fig,ax=plt.subplots(figsize=(6,4))
ax.scatter(x_reg,y_reg,s=5,alpha=0.4); ax.plot(x_reg,yp,"r",linewidth=2,label="NN fit")
ax.plot(x_reg,0.5*x_reg**2,"g--",label="True"); ax.legend(); ax.set_title("NN Regression")
plt.savefig("nn_regression.png",dpi=72); plt.close(); print("Saved nn_regression.png, activations.png")

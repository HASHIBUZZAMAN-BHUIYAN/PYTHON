# Advanced Day 10 — Solutions
import os; os.environ["TF_CPP_MIN_LOG_LEVEL"]="2"
import numpy as np, matplotlib.pyplot as plt, tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.datasets import make_regression, load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
np.random.seed(42); tf.random.set_seed(42)

# Shared digits data
digits=load_digits(); sc=StandardScaler()
X=sc.fit_transform(digits.data).astype(np.float32); y=digits.target
Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=0.2,stratify=y,random_state=42)

# Exercise 1
Xr,yr=make_regression(500,n_features=10,noise=10,random_state=42)
Xr=StandardScaler().fit_transform(Xr).astype(np.float32); yr=yr.astype(np.float32)
Xrtr,Xrte,yrtr,yrte=train_test_split(Xr,yr,test_size=0.2,random_state=42)
reg=keras.Sequential([layers.Input(10),layers.Dense(64,"relu"),layers.Dense(32,"relu"),layers.Dense(1)])
reg.compile("adam","mse",["mae"])
h=reg.fit(Xrtr,yrtr,epochs=20,validation_data=(Xrte,yrte),batch_size=32,verbose=0)
print(f"Regression MAE: {reg.evaluate(Xrte,yrte,verbose=0)[1]:.2f}")

# Exercise 2 — Custom training loop
m=keras.Sequential([layers.Input(64),layers.Dense(128,"relu"),layers.Dense(64,"relu"),layers.Dense(10,"softmax")])
opt=keras.optimizers.Adam(1e-3); loss_fn=keras.losses.SparseCategoricalCrossentropy()
ds=tf.data.Dataset.from_tensor_slices((Xtr,ytr)).batch(32)
for ep in range(10):
    total=0.
    for xb,yb in ds:
        with tf.GradientTape() as tape:
            l=loss_fn(yb,m(xb,training=True))
        grads=tape.gradient(l,m.trainable_variables)
        opt.apply_gradients(zip(grads,m.trainable_variables)); total+=float(l)
    print(f"Ep {ep+1}: {total/len(ds):.4f}")

# Exercise 3
fig,ax=plt.subplots(figsize=(8,4))
for label,dropout,reg in [("None",0,None),("Dropout",0.5,None),("L2",0,keras.regularizers.l2(0.001))]:
    m2=keras.Sequential([layers.Input(64),
        layers.Dense(128,"relu",kernel_regularizer=reg),
        layers.Dropout(dropout) if dropout>0 else layers.Lambda(lambda x:x),
        layers.Dense(128,"relu",kernel_regularizer=reg),
        layers.Dropout(dropout) if dropout>0 else layers.Lambda(lambda x:x),
        layers.Dense(10,"softmax")])
    m2.compile("adam","sparse_categorical_crossentropy",["accuracy"])
    h=m2.fit(Xtr,ytr,20,validation_data=(Xte,yte),batch_size=32,verbose=0)
    ax.plot(h.history["val_accuracy"],label=label)
ax.legend(); ax.set_title("Val accuracy with regularization"); plt.savefig("reg_compare.png",dpi=72); plt.close()

# Exercise 4
class MyModel(keras.Model):
    def __init__(self):
        super().__init__()
        self.d1=layers.Dense(128,activation="relu"); self.d2=layers.Dense(64,activation="relu"); self.d3=layers.Dense(10,activation="softmax")
    def call(self,x): return self.d3(self.d2(self.d1(x)))

m3=MyModel(); m3.compile("adam","sparse_categorical_crossentropy",["accuracy"])
m3.fit(Xtr,ytr,epochs=10,validation_data=(Xte,yte),batch_size=32,verbose=0)
print(f"Subclass model test acc: {m3.evaluate(Xte,yte,verbose=0)[1]:.4f}")

# Exercise 5
steps=500; sched=keras.optimizers.schedules.CosineDecay(1e-3,steps)
lrs=[float(sched(i)) for i in range(steps)]
plt.plot(lrs); plt.title("Cosine LR schedule"); plt.savefig("lr_schedule.png",dpi=72); plt.close()
print("Saved reg_compare.png, lr_schedule.png")

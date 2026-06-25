# Advanced Day 07 — Solutions
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits, make_classification
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, validation_curve
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, classification_report)
np.random.seed(42)

digits = load_digits()
idx = np.random.choice(len(digits.data), 500, replace=False)
X,y = digits.data[idx], digits.target[idx]
Xtr,Xte,ytr,yte = train_test_split(X,y,test_size=0.3,random_state=42,stratify=y)

# Exercise 1
pipe_rf = Pipeline([("sc",StandardScaler()),("rf",RandomForestClassifier(50,random_state=42))])
pipe_rf.fit(Xtr,ytr); yp = pipe_rf.predict(Xte)
print(classification_report(yte,yp))

# Exercise 2
y_bin = (y==5).astype(int)
Xtr2,Xte2,ytr2,yte2 = train_test_split(X,y_bin,test_size=0.3,random_state=42)
lr = Pipeline([("sc",StandardScaler()),("lr",LogisticRegression(max_iter=1000))])
lr.fit(Xtr2,ytr2); probs = lr.predict_proba(Xte2)[:,1]
for th in [0.3,0.4,0.5,0.6,0.7]:
    pred = (probs>=th).astype(int)
    print(f"th={th}: P={precision_score(yte2,pred,zero_division=0):.3f} R={recall_score(yte2,pred,zero_division=0):.3f}")

# Exercise 3
X_imb,y_imb = make_classification(1000,weights=[0.9,0.1],random_state=42)
Xtr3,Xte3,ytr3,yte3 = train_test_split(X_imb,y_imb,test_size=0.3,random_state=42)
for cw in [None,"balanced"]:
    m = LogisticRegression(class_weight=cw,max_iter=1000).fit(Xtr3,ytr3); p=m.predict(Xte3)
    print(f"class_weight={cw}: acc={accuracy_score(yte3,p):.3f} f1={f1_score(yte3,p):.3f}")

# Exercise 4
param_grid={"rf__n_estimators":[10,50],"rf__max_depth":[3,5,None]}
pipe2=Pipeline([("sc",StandardScaler()),("rf",RandomForestClassifier(random_state=42))])
gs=GridSearchCV(pipe2,param_grid,cv=5,n_jobs=-1)
gs.fit(Xtr,ytr); print(gs.best_params_,gs.best_score_,accuracy_score(yte,gs.predict(Xte)))

# Exercise 5
depths=range(1,16)
tr_sc,cv_sc=[],[]
for d in depths:
    m=Pipeline([("sc",StandardScaler()),("rf",RandomForestClassifier(20,max_depth=d,random_state=42))])
    cv=cross_val_score(m,Xtr,ytr,cv=5); cv_sc.append(cv.mean())
    m.fit(Xtr,ytr); tr_sc.append(accuracy_score(ytr,m.predict(Xtr)))
fig,ax=plt.subplots(figsize=(8,4))
ax.plot(depths,tr_sc,label="Train"); ax.plot(depths,cv_sc,label="CV Val")
ax.set_title("Validation Curve"); ax.legend(); plt.savefig("val_curve.png",dpi=72); plt.close()
print("Saved val_curve.png")

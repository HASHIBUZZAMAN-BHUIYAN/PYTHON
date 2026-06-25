# Advanced Day 06 — Solutions
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer, load_wine
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, roc_auc_score
np.random.seed(42)

bc = load_breast_cancer(); X,y = bc.data, bc.target
Xtr,Xte,ytr,yte = train_test_split(X,y,test_size=0.2,random_state=42)
sc = StandardScaler(); Xtr_s = sc.fit_transform(Xtr); Xte_s = sc.transform(Xte)

# Exercise 1
for name,m,scaled in [("DT",DecisionTreeClassifier(random_state=42),False),
                       ("RF",RandomForestClassifier(50,random_state=42),False),
                       ("KNN",KNeighborsClassifier(7),True),
                       ("SVM",SVC(probability=True,random_state=42),True)]:
    Xt,Xe = (Xtr_s,Xte_s) if scaled else (Xtr,Xte)
    m.fit(Xt,ytr); p = m.predict(Xe)
    auc = roc_auc_score(yte, m.predict_proba(Xe)[:,1] if hasattr(m,"predict_proba") else m.decision_function(Xe))
    print(f"{name}: acc={accuracy_score(yte,p):.4f} auc={auc:.4f}")

# Exercise 2 + 3
fig,axes = plt.subplots(1,2,figsize=(12,4))
tr_accs=[]; te_accs=[]; ns=range(1,101,10)
for n in ns:
    m=RandomForestClassifier(n,random_state=42).fit(Xtr,ytr)
    tr_accs.append(accuracy_score(ytr,m.predict(Xtr))); te_accs.append(accuracy_score(yte,m.predict(Xte)))
axes[0].plot(ns,tr_accs,label="Train"); axes[0].plot(ns,te_accs,label="Test")
axes[0].set_title("RF: n_estimators vs Accuracy"); axes[0].legend()

rf100 = RandomForestClassifier(100,random_state=42).fit(Xtr,ytr)
imp = sorted(zip(bc.feature_names,rf100.feature_importances_),key=lambda x:-x[1])[:10]
axes[1].barh([i[0] for i in imp],[i[1] for i in imp],color="steelblue")
axes[1].set_title("Top 10 Features"); plt.tight_layout(); plt.savefig("ex_d06.png",dpi=72); plt.close()

# Exercise 4
wine = load_wine(); Xw,yw = wine.data,wine.target
sc2 = StandardScaler()
for name,m in [("DT",DecisionTreeClassifier(random_state=42)),("RF",RandomForestClassifier(50,random_state=42)),
               ("KNN",KNeighborsClassifier(5)),("SVM",SVC(random_state=42))]:
    Xs = sc2.fit_transform(Xw) if name in ("KNN","SVM") else Xw
    cv = cross_val_score(m,Xs,yw,cv=10); print(f"{name}: {cv.mean():.4f} ± {cv.std():.4f}")

# Exercise 5
tr_accs=[]; te_accs=[]; depths=range(1,21)
for d in depths:
    m=DecisionTreeClassifier(max_depth=d,random_state=42).fit(Xtr,ytr)
    tr_accs.append(accuracy_score(ytr,m.predict(Xtr))); te_accs.append(accuracy_score(yte,m.predict(Xte)))
fig,ax=plt.subplots(figsize=(8,4))
ax.plot(depths,tr_accs,label="Train"); ax.plot(depths,te_accs,label="Test")
ax.axvline(depths[np.argmax(te_accs)],color="red",linestyle="--",label=f"Best depth={depths[np.argmax(te_accs)]}")
ax.legend(); ax.set_title("DT: Overfitting curve"); plt.savefig("overfitting.png",dpi=72); plt.close()
print("Saved ex_d06.png, overfitting.png")

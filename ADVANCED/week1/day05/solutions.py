# Advanced Day 05 — Solutions
import numpy as np
from sklearn.datasets import load_wine, make_regression, make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso
from sklearn.metrics import accuracy_score, r2_score, mean_squared_error, classification_report, confusion_matrix
np.random.seed(42)

# Exercise 1
X, y = make_regression(300, n_features=5, noise=10, random_state=42)
Xtr,Xte,ytr,yte = train_test_split(X,y,test_size=0.2,random_state=42)
m = LinearRegression().fit(Xtr,ytr); yp = m.predict(Xte)
print(f"R²={r2_score(yte,yp):.4f} RMSE={mean_squared_error(yte,yp)**0.5:.2f}")
print("Coefs:", m.coef_.round(2))
print("Most important feature:", np.argmax(np.abs(m.coef_)))

# Exercise 2
wine = load_wine(); X,y = wine.data, wine.target
Xtr,Xte,ytr,yte = train_test_split(X,y,test_size=0.25,random_state=42)
sc = StandardScaler(); Xtr_s = sc.fit_transform(Xtr); Xte_s = sc.transform(Xte)
clf = LogisticRegression(max_iter=2000,random_state=42).fit(Xtr_s,ytr)
yp = clf.predict(Xte_s)
print(f"\nWine accuracy: {accuracy_score(yte,yp):.4f}")
print(classification_report(yte,yp,target_names=wine.target_names))
print(confusion_matrix(yte,yp))

# Exercise 3
X,y = make_classification(200,n_features=5,random_state=42)
sc = StandardScaler()
for ts in [0.1,0.2,0.3,0.4,0.5]:
    Xtr,Xte,ytr,yte = train_test_split(X,y,test_size=ts,random_state=42)
    clf = LogisticRegression(max_iter=500).fit(sc.fit_transform(Xtr),ytr)
    tr_acc = accuracy_score(ytr, clf.predict(sc.transform(Xtr)))
    te_acc = accuracy_score(yte, clf.predict(sc.transform(Xte)))
    print(f"test_size={ts}: train_acc={tr_acc:.3f}, test_acc={te_acc:.3f}")

# Exercise 4
x = np.linspace(-3,3,100).reshape(-1,1)
y = 3*x.flatten()**2 - 2*x.flatten() + 1 + np.random.normal(0,1,100)
Xtr,Xte,ytr,yte = train_test_split(x,y,test_size=0.2,random_state=42)
m1 = LinearRegression().fit(Xtr,ytr)
X_poly = np.column_stack([x,x**2]); Xtr2,Xte2,_,_ = train_test_split(X_poly,y,test_size=0.2,random_state=42)
m2 = LinearRegression().fit(Xtr2,ytr)
print(f"\nLinear R²={r2_score(yte,m1.predict(Xte)):.3f}")
print(f"Quadratic R²={r2_score(yte,m2.predict(Xte2)):.3f}")

# Exercise 5
X,y = make_regression(200,n_features=10,n_informative=3,noise=5,random_state=42)
Xtr,Xte,ytr,yte = train_test_split(X,y,test_size=0.2,random_state=42)
for name,m in [("LinearReg",LinearRegression()),("Ridge1",Ridge(1)),("Ridge100",Ridge(100)),("Lasso",Lasso(0.1))]:
    m.fit(Xtr,ytr); print(f"{name}: R²={r2_score(yte,m.predict(Xte)):.4f} coefs={m.coef_.round(1)}")

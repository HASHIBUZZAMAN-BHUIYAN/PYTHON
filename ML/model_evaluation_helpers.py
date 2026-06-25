# ML Reference — Model Evaluation Helpers
# Import and call these functions for any classification task.

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix, classification_report,
    roc_curve, auc, precision_recall_curve, average_precision_score,
)
from sklearn.model_selection import learning_curve, cross_val_score

# ─── CONFUSION MATRIX ────────────────────────────────────────────────────────
def plot_confusion_matrix(y_true, y_pred, class_names=None, title="Confusion Matrix", ax=None):
    cm = confusion_matrix(y_true, y_pred)
    if ax is None: fig, ax = plt.subplots(figsize=(6,5))
    im = ax.imshow(cm, cmap="Blues")
    plt.colorbar(im, ax=ax)
    if class_names is None: class_names = [str(i) for i in range(cm.shape[0])]
    ax.set_xticks(range(len(class_names))); ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticks(range(len(class_names))); ax.set_yticklabels(class_names)
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i,j]), ha="center", va="center",
                    color="white" if cm[i,j]>thresh else "black")
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(title)
    return cm

# ─── CLASSIFICATION REPORT ───────────────────────────────────────────────────
def print_classification_report(y_true, y_pred, class_names=None):
    print(classification_report(y_true, y_pred, target_names=class_names))

# ─── ROC CURVE ────────────────────────────────────────────────────────────────
def plot_roc_curve(y_true, y_scores, label="Model", ax=None):
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    if ax is None: fig, ax = plt.subplots(figsize=(5,5))
    ax.plot(fpr, tpr, linewidth=2, label=f"{label} (AUC={roc_auc:.3f})")
    ax.plot([0,1],[0,1],"k--",linewidth=1)
    ax.set_xlabel("FPR"); ax.set_ylabel("TPR"); ax.set_title("ROC Curve")
    ax.legend(); ax.grid(alpha=0.3)
    return roc_auc

def plot_roc_multi(classifiers_dict, X_test, y_test, pos_label=1, ax=None):
    """classifiers_dict: {name: fitted_clf}"""
    if ax is None: fig, ax = plt.subplots(figsize=(6,5))
    for name, clf in classifiers_dict.items():
        probs = clf.predict_proba(X_test)[:,pos_label]
        fpr,tpr,_ = roc_curve(y_test, probs)
        auc_val = auc(fpr,tpr)
        ax.plot(fpr,tpr,linewidth=2,label=f"{name} (AUC={auc_val:.3f})")
    ax.plot([0,1],[0,1],"k--",linewidth=1)
    ax.set_xlabel("FPR"); ax.set_ylabel("TPR"); ax.set_title("ROC Curves")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

# ─── PRECISION-RECALL CURVE ──────────────────────────────────────────────────
def plot_pr_curve(y_true, y_scores, label="Model", ax=None):
    prec, rec, _ = precision_recall_curve(y_true, y_scores)
    ap = average_precision_score(y_true, y_scores)
    if ax is None: fig, ax = plt.subplots(figsize=(5,5))
    ax.plot(rec, prec, linewidth=2, label=f"{label} (AP={ap:.3f})")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve"); ax.legend(); ax.grid(alpha=0.3)
    return ap

# ─── LEARNING CURVE ──────────────────────────────────────────────────────────
def plot_learning_curve(estimator, X, y, cv=5, scoring="accuracy", ax=None):
    train_sizes, train_scores, val_scores = learning_curve(
        estimator, X, y, cv=cv, scoring=scoring,
        train_sizes=np.linspace(0.1, 1.0, 8), n_jobs=-1
    )
    if ax is None: fig, ax = plt.subplots(figsize=(7,4))
    ax.plot(train_sizes, train_scores.mean(1), "b-o", label="Train")
    ax.fill_between(train_sizes,
                    train_scores.mean(1)-train_scores.std(1),
                    train_scores.mean(1)+train_scores.std(1), alpha=0.1, color="b")
    ax.plot(train_sizes, val_scores.mean(1), "r-o", label="Validation")
    ax.fill_between(train_sizes,
                    val_scores.mean(1)-val_scores.std(1),
                    val_scores.mean(1)+val_scores.std(1), alpha=0.1, color="r")
    ax.set_xlabel("Training set size"); ax.set_ylabel(scoring)
    ax.set_title("Learning Curve"); ax.legend(); ax.grid(alpha=0.3)

# ─── CROSS-VALIDATION SUMMARY ────────────────────────────────────────────────
def cv_summary(classifiers_dict, X, y, cv=5, scoring="accuracy"):
    print(f"{'Model':<25} {'Mean':>8} {'Std':>7}")
    print("-"*42)
    for name, clf in classifiers_dict.items():
        scores = cross_val_score(clf, X, y, cv=cv, scoring=scoring)
        print(f"{name:<25} {scores.mean():>8.4f} {scores.std():>7.4f}")

# ─── DEMO ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from sklearn.datasets import make_classification
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split

    X, y = make_classification(n_samples=500, n_features=10, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

    lr  = LogisticRegression(max_iter=1000).fit(X_train, y_train)
    rf  = RandomForestClassifier(n_estimators=50, random_state=42).fit(X_train, y_train)

    y_pred = lr.predict(X_test)
    y_prob = lr.predict_proba(X_test)[:,1]

    fig, axes = plt.subplots(1, 3, figsize=(15,4))
    plot_confusion_matrix(y_test, y_pred, ax=axes[0])
    plot_roc_curve(y_test, y_prob, "Logistic Regression", ax=axes[1])
    plot_pr_curve(y_test, y_prob, "Logistic Regression", ax=axes[2])
    plt.tight_layout(); plt.savefig("eval_demo.png", dpi=80)
    print_classification_report(y_test, y_pred)
    cv_summary({"LR": lr, "RF": rf}, X, y)
    print("Saved eval_demo.png")

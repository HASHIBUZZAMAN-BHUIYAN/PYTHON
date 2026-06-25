# Advanced Day 24 — Solutions
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report

POS = ["wonderful","loved","fantastic","amazing","great","excellent","brilliant","outstanding",
       "best","recommend","happy","perfect","masterpiece","superb","delightful","enjoyable",
       "impressive","splendid","terrific","marvelous","exceptional","pleasant","glorious",
       "joyful","beautiful"]
NEG = ["terrible","awful","worst","poor","horrible","bad","dreadful","disgusting","garbage",
       "disappointed","useless","damaged","broke","avoid","waste","dreadful","abysmal",
       "pathetic","dismal","atrocious","mediocre","dull","boring","frustrating","unacceptable"]

POS_R = ["This movie was absolutely wonderful.",
         "Fantastic product, highly recommend.",
         "Amazing performance, the best ever.",
         "Great value, very happy with my purchase.",
         "Excellent quality, outstanding service."] * 6  # 30 pos
NEG_R = ["Terrible, broke after one day.",
         "Awful experience, customer service useless.",
         "Worst movie ever, complete waste.",
         "Poor quality, very disappointed.",
         "Horrible, arrived damaged."] * 6  # 30 neg
NEU_R = ["The package arrived on Tuesday.",
         "The product is 15 cm wide.",
         "I ordered the blue version.",
         "It comes with a user manual.",
         "The return policy is 30 days."] * 6  # 30 neutral

# Ex 1
print("=== Ex 1: Expanded Lexicon ===")
def lex_3class(text):
    words=set(text.lower().split())
    p=len(words&set(POS)); n=len(words&set(NEG))
    if p>n: return 1
    if n>p: return 0
    return 2
labels_3=[1]*30+[0]*30+[2]*30
reviews_3=POS_R+NEG_R+NEU_R
preds=[lex_3class(r) for r in reviews_3]
acc=np.mean(np.array(preds)==np.array(labels_3))
print(f"  3-class lexicon accuracy: {acc:.1%}")

# Ex 2
print("\n=== Ex 2: Trigram TF-IDF ===")
for ng in [(1,2),(1,3)]:
    v=TfidfVectorizer(ngram_range=ng,max_features=1000)
    X=v.fit_transform(POS_R[:20]+NEG_R[:20])
    y=np.array([1]*20+[0]*20)
    lr=LogisticRegression(max_iter=500)
    sc=cross_val_score(lr,X,y,cv=4).mean()
    print(f"  ngram={ng}: CV acc={sc:.3f}")

# Ex 3
print("\n=== Ex 3: 3-Class Classification ===")
vec3=TfidfVectorizer(max_features=500)
X3=vec3.fit_transform(reviews_3)
y3=np.array(labels_3)
lr3=LogisticRegression(max_iter=1000)
lr3.fit(X3,y3)
from sklearn.model_selection import cross_val_predict
ypred=cross_val_predict(lr3,X3,y3,cv=4)
print(classification_report(y3,ypred,target_names=["NEGATIVE","POSITIVE","NEUTRAL"],zero_division=0))

# Ex 4
print("=== Ex 4: Confidence Threshold ===")
X2=TfidfVectorizer(max_features=500).fit_transform(POS_R[:15]+NEG_R[:15])
y2=np.array([1]*15+[0]*15)
lr4=LogisticRegression(max_iter=1000).fit(X2,y2)
probs=lr4.predict_proba(X2)[:,1]
certain=np.sum((probs>=0.7)|(probs<=0.3))
print(f"  Certain (>=0.7 confidence): {certain}/{len(probs)} = {certain/len(probs):.0%}")

# Ex 5
print("\n=== Ex 5: Domain Adaptation ===")
movie_pos=["Brilliant film, stunning performances.","Amazing cinematography, must watch.",
           "Best movie of the year.","Masterpiece of modern cinema."]*4
movie_neg=["Boring plot, terrible acting.","Worst film ever made.","Complete waste.","Dreadful."]*4
prod_pos =["Great product, works perfectly.","Excellent quality, highly recommend.",
           "Perfect fit, love it.","Outstanding value."]*4
prod_neg =["Broke after one day.","Poor quality, very disappointed.",
           "Arrived damaged.","Terrible product."]*4
labels_m=[1]*16+[0]*16; labels_p=[1]*16+[0]*16
vec5=TfidfVectorizer(max_features=300)
Xm=vec5.fit_transform(movie_pos+movie_neg)
Xp=vec5.transform(prod_pos+prod_neg)
lr5=LogisticRegression(max_iter=500).fit(Xm,labels_m)
cross_acc=np.mean(lr5.predict(Xp)==labels_p)
vec6=TfidfVectorizer(max_features=300).fit(prod_pos+prod_neg)
Xp2=vec6.transform(prod_pos+prod_neg)
in_acc=cross_val_score(LogisticRegression(max_iter=500),Xp2,labels_p,cv=4).mean()
print(f"  Cross-domain acc : {cross_acc:.1%}")
print(f"  In-domain acc    : {in_acc:.1%}")
print(f"  Domain shift cost: {in_acc-cross_acc:.1%}")

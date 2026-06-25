# Advanced Day 28 — Solutions
import re, math
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import deque

KB = [
    ("what is machine learning",   "ML is a type of AI where systems learn from data."),
    ("how does deep learning work","Deep learning uses multi-layer neural networks."),
    ("what is python",             "Python is a high-level versatile programming language."),
    ("what is overfitting",        "Overfitting is when a model memorizes training data too well."),
    ("what is gradient descent",   "Gradient descent minimizes the loss by adjusting weights iteratively."),
    ("what is nlp",                "NLP enables computers to understand human language."),
    ("what is a transformer",      "Transformers use self-attention to process sequences in parallel."),
    ("what is regularization",     "Regularization adds penalty terms to prevent overfitting."),
    ("what is a neural network",   "Neural networks are interconnected layers of computational nodes."),
    ("what is supervised learning","Supervised learning trains models using labeled (input, output) pairs."),
]

def preprocess(t): return re.sub(r"[^\w\s]","",t.lower()).strip()
questions = [preprocess(q) for q,_ in KB]
answers   = [a for _,a in KB]
vec = TfidfVectorizer(ngram_range=(1,2),max_features=500).fit(questions)
Qmat= vec.transform(questions)
def chat(q, threshold=0.1):
    v=vec.transform([preprocess(q)])
    s=cosine_similarity(v,Qmat).flatten()
    b=int(np.argmax(s))
    return (answers[b],float(s[b])) if s[b]>=threshold else ("I don't know.",0.0)

# Ex 1
print("=== Ex 1: Expanded KB ===")
NEW_KB = [
    ("what is reinforcement learning","RL trains agents to maximize cumulative reward via trial and error."),
    ("what is backpropagation",       "Backprop computes gradients via chain rule to update weights."),
    ("what is dropout",               "Dropout randomly zeros neurons during training as regularization."),
    ("what is batch normalization",   "BatchNorm normalizes layer inputs for faster, stable training."),
    ("what is transfer learning",     "Transfer learning reuses a pretrained model on a new task."),
]
all_kb = KB + NEW_KB
qs2=[preprocess(q) for q,_ in all_kb]; as2=[a for _,a in all_kb]
v2=TfidfVectorizer(ngram_range=(1,2),max_features=800).fit(qs2)
Q2=v2.transform(qs2)
def chat2(q,t=0.1):
    s=cosine_similarity(v2.transform([preprocess(q)]),Q2).flatten(); b=int(np.argmax(s))
    return (as2[b],float(s[b])) if s[b]>=t else ("I don't know.",0.0)
paraphrases=[
    ("how does RL work?","reinforcement learning"),
    ("what is backprop?","backpropagation"),
    ("explain dropout","dropout"),
]
for q,topic in paraphrases:
    a,c=chat2(q); print(f"  [{c:.2f}] {q} → {a[:60]}")

# Ex 2
print("\n=== Ex 2: BM25 ===")
def bm25_score(query_toks, docs, k=1.5, b=0.75):
    avgdl = np.mean([len(d.split()) for d in docs])
    N     = len(docs)
    scores= np.zeros(N)
    for t in query_toks:
        df   = sum(1 for d in docs if t in d.split())
        idf  = math.log((N - df + 0.5) / (df + 0.5) + 1)
        for i,doc in enumerate(docs):
            tf   = doc.split().count(t)
            dl   = len(doc.split())
            scores[i] += idf * (tf*(k+1)) / (tf + k*(1-b+b*dl/avgdl))
    return scores

test_queries=[("explain machine learning","what is machine learning"),
              ("how python works","what is python"),
              ("what is dropout","what is dropout")]
for q,expected in test_queries:
    q_toks = preprocess(q).split()
    bm25  = int(np.argmax(bm25_score(q_toks, qs2)))
    tfidf = int(np.argmax(cosine_similarity(v2.transform([preprocess(q)]),Q2).flatten()))
    print(f"  BM25={qs2[bm25][:30]}  TF-IDF={qs2[tfidf][:30]}  expected={expected}")

# Ex 3
print("\n=== Ex 3: Multi-Turn Memory ===")
history = deque(maxlen=3)
def chat_with_history(query):
    ctx    = " ".join(a for _,a in list(history))
    full_q = preprocess(query + " " + ctx)
    s      = cosine_similarity(v2.transform([full_q]),Q2).flatten()
    b      = int(np.argmax(s))
    ans    = as2[b] if s[b]>=0.1 else "I don't know."
    history.append((query, ans))
    return ans, float(s[b])
for q in ["what is machine learning","explain that further","what else can you tell me"]:
    a,c=chat_with_history(q); print(f"  [{c:.3f}] {q} → {a[:60]}")

# Ex 4
print("\n=== Ex 4: Confidence Calibration ===")
in_kb  = [preprocess(q) for q,_ in KB]
ood    = ["what is pizza","who is the best actor","how do I cook rice",
          "what time is it","how tall is mount everest","best book ever written",
          "what is the speed of light","how to make coffee","who invented tv","what year is it"]
in_scores =[float(cosine_similarity(v2.transform([q]),Q2).flatten().max()) for q in in_kb]
ood_scores=[float(cosine_similarity(v2.transform([preprocess(q)]),Q2).flatten().max()) for q in ood]
best_t = max(np.arange(0.05,0.9,0.05),
             key=lambda t: sum(s>=t for s in in_scores)-sum(s>=t for s in ood_scores))
fig,ax=plt.subplots(figsize=(7,3))
ax.hist(in_scores,bins=10,alpha=0.7,label="In-KB",color="green")
ax.hist(ood_scores,bins=10,alpha=0.7,label="Out-of-domain",color="red")
ax.axvline(best_t,color="k",linestyle="--",label=f"Best threshold={best_t:.2f}")
ax.legend(); ax.set_xlabel("Cosine similarity"); ax.set_title("Confidence Calibration")
plt.tight_layout(); plt.savefig("confidence.png",dpi=80); plt.close()
print(f"  Best threshold: {best_t:.2f}  → saved confidence.png")

# Ex 5
print("\n=== Ex 5: Response Templating ===")
TEMPLATES=["Great question about {topic}! {answer}","Regarding {topic}: {answer}","About {topic} — {answer}"]
def extract_topic(query):
    words=re.findall(r"[a-z]+",query.lower())
    stopwords={"what","is","are","a","an","the","how","does","do","can","you","tell","me","about","explain"}
    content=[w for w in words if w not in stopwords]
    return " ".join(content[:2]) if content else "your question"
import random; random.seed(42)
for q in ["what is deep learning","explain gradient descent","what is overfitting"]:
    a,c=chat(q)
    topic=extract_topic(q)
    resp=random.choice(TEMPLATES).format(topic=topic,answer=a)
    print(f"  {resp[:90]}")

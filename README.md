# Python Learning Curriculum — Complete Guide

> **Hardware:** AMD Ryzen 7 5000-series · 8 GB RAM · **No GPU** — all lessons run CPU-only.

---

## Quick Start

```powershell
# Activate BEGINNER virtual environment
.\BEGINNER\.venv\Scripts\Activate.ps1
python BEGINNER\day01\lesson.py

# Activate ADVANCED virtual environment
.\ADVANCED\.venv\Scripts\Activate.ps1
python ADVANCED\week1\day01\lesson.py
```

See `SETUP.md` for full setup and reinstall instructions.

---

## Curriculum Map

### BEGINNER Track (14 days) — `BEGINNER\.venv`

| Day | Topic | Mini-Project |
|-----|-------|-------------|
| 01 | Variables, data types, I/O | Personal Profile Card Generator |
| 02 | Operators, expressions, type casting | Bill Splitter |
| 03 | Conditionals | Grade Report Generator |
| 04 | Loops | Multiplication Table Printer |
| 05 | Strings | Text Analyzer |
| 06 | Lists & Tuples | Student Score Manager |
| 07 | Dicts & Sets | Inventory System |
| 08 | Functions, args/kwargs, recursion | Math Toolkit |
| 09 | File handling, CSV | CSV Journal |
| 10 | Error handling, custom exceptions | Robust Calculator |
| 11 | OOP I — classes, `__init__`, `__str__` | Library Book System |
| 12 | OOP II — inheritance, polymorphism, ABC | RPG Character System |
| 13 | Modules, packages, pip, APIs | Weather CLI (Open-Meteo) |
| 14 | **CAPSTONE** — Contact Book CLI | Full CLI app (CSV + OOP) |

---

### ADVANCED Track (42 days) — `ADVANCED\.venv`

#### Week 1 — Data Science

| Day | Topic | Mini-Project |
|-----|-------|-------------|
| 01 | NumPy — arrays, broadcasting, linear algebra | Image processing (synthetic) |
| 02 | Pandas — DataFrames, groupby, merge | Student Performance Analysis |
| 03 | Matplotlib / Seaborn visualization | Sales Dashboard (4-panel) |
| 04 | Statistics & probability (scipy.stats) | Health Data Statistical Report |
| 05 | Intro to ML (scikit-learn) | House Price Predictor |
| 06 | Classical ML — DT, RF, KNN, SVM, GridSearch | Spam Classifier |
| 07 | Model evaluation — confusion matrix, ROC, PR, learning curve | Churn Prediction pipeline |

#### Week 2 — Deep Learning

| Day | Topic | Mini-Project |
|-----|-------|-------------|
| 08 | Neural network from scratch (NumPy backprop) | Digits 0-vs-1 classifier |
| 09 | PyTorch — tensors, autograd, nn.Module, DataLoader | Multi-class digits |
| 10 | TensorFlow/Keras — Sequential/Functional API, callbacks | Keras vs PyTorch benchmark |
| 11 | CNN fundamentals — Conv2d, BN, MaxPool, feature maps | MNIST CNN (5000 samples) |
| 12 | CNN tuning — augmentation, BN, Dropout, LR scheduling | Ablation study |
| 13 | Transfer learning — MobileNetV2 frozen backbone | Shapes classifier (3-class) |
| 14 | **CAPSTONE Week 2** — 5-class shapes CNN | Full confusion matrix report |

#### Week 3 — Robotics & Agentic AI

| Day | Topic | Mini-Project |
|-----|-------|-------------|
| 15 | Robotics foundations — FK/IK, differential drive | Animated figure-8 arm (GIF) |
| 16 | PID controller — P/I/D terms, tuning, anti-windup | Curved track following |
| 17 | Path planning — BFS, A*, Q-learning | Multi-room navigation |
| 18 | Computer vision for robotics — OpenCV | Color-guided robot navigation |
| 19 | Agentic AI — rule-based FSM agents | Warehouse robot agent |
| 20 | LLM-powered agents — ReAct, tool calling | Research assistant agent |
| 21 | **CAPSTONE Week 3** — Full agentic robotic system | Autonomous rescue mission |

---

#### Week 4 — NLP Track *(projects/ folder: 3 runnable projects per day)*

| Day | Topic | Key Concepts |
|-----|-------|-------------|
| 22 | NLP foundations — tokenization, stemming, TF-IDF | NLTK, Bag-of-Words |
| 23 | Word embeddings — Word2Vec, GloVe analogy | Gensim, cosine similarity |
| 24 | Sentiment analysis — lexicon, TF-IDF+LR, HuggingFace distilbert | 3-approach comparison |
| 25 | NER & POS tagging — regex patterns, NLTK, spaCy | Entity extraction |
| 26 | Text classification — BoW+NB, TF-IDF+LR, zero-shot | 4-class news classifier |
| 27 | Text generation — Markov chains, extractive summarization | Bigram/trigram models |
| 28 | Retrieval chatbot — TF-IDF cosine similarity, MRR/Hit@K | Confidence threshold |

#### Week 5 — Advanced Deep Learning *(projects/ folder: 3 runnable projects per day)*

| Day | Topic | Key Concepts |
|-----|-------|-------------|
| 29 | Advanced PyTorch — custom datasets, hooks, mixed precision | DataLoader, LR finder |
| 30 | Regularization — dropout, weight decay, label smoothing | Generalization techniques |
| 31 | Autoencoders — encoder-decoder, denoising AE, anomaly detection | MSE loss, 95th percentile threshold |
| 32 | GANs — adversarial training, WGAN, conditional GAN | BCE loss, mode collapse |
| 33 | Attention from scratch — scaled dot-product, causal mask, multi-head | NumPy + PyTorch |
| 34 | Transformers — PositionalEncoding, TransformerBlock, TinyTransformer | Sinusoidal embeddings |
| 35 | Architecture comparison — RNN/LSTM/GRU/BiLSTM/Transformer benchmark | Pareto analysis |

#### Week 6 — Agentic AI *(projects/ folder: 3 runnable projects per day)*

| Day | Topic | Key Concepts |
|-----|-------|-------------|
| 36 | Agent foundations review | ReAct, FSM, tool calling recap |
| 37 | Planning agents — Plan-then-Execute, task dependency graphs | Topological sort execution |
| 38 | Tool systems — typed registry, retry/backoff, fallbacks, observability | Rate limiting, parallel tools |
| 39 | Multi-agent systems — pipeline, debate, parallel, negotiation | Message protocol, judge agent |
| 40 | Agent evaluation & guardrails — PII detection, safety checks, BLEU | Adversarial input testing |
| 41 | Agents + simulation — grid world, A*/BFS navigation, multi-robot | Reactive vs deliberative |
| 42 | **FINAL CAPSTONE** — End-to-end AI assistant | NLP + DL + Agent + Guardrails integrated |

---

### Reference Folders

| Folder | Files | Purpose |
|--------|-------|---------|
| `ML/`         | `linear_regression_template.py`, `model_evaluation_helpers.py` | ML pipelines & evaluation |
| `DL/`         | `pytorch_training_loop_template.py`, `keras_training_loop_template.py`, `autoencoder_template.py`, `gan_template.py`, `transformer_template.py` | DL training loops + architectures |
| `CNN/`        | `cnn_architectures_cheatsheet.py` | LeNet, VGG-mini, ResNet, MobileNet, transfer |
| `ROBOTICS/`   | `pid_controller.py`, `pathfinding_astar.py` | PID (1D/2D/cascade), A*/BFS/Dijkstra |
| `NLP/`        | `sentiment_pipeline.py`, `ner_pipeline.py`, `text_classifier.py`, `text_generator.py`, `retrieval_chatbot.py` | NLP pipelines (all CPU, offline fallbacks) |
| `AGENTIC_AI/` | `simple_agent_template.py`, `llm_tool_calling_template.py`, `planning_agent_template.py`, `multi_agent_template.py` | Reflex/FSM/Goal agents, ReAct, multi-agent |

---

## Directory Structure

```
PYTHON/
├── BEGINNER/
│   ├── .venv/                   ← stdlib + matplotlib + requests
│   ├── day01/ ... day14/
│   │   ├── README.md, lesson.py, exercises.py, solutions.py, mini_project.py
│   └── README.md
├── ADVANCED/
│   ├── .venv/                   ← numpy pandas matplotlib sklearn torch(cpu) tensorflow-cpu opencv spacy nltk
│   ├── week1/day01/ ... day07/  ← Data science
│   ├── week2/day08/ ... day14/  ← CNN & Deep learning
│   ├── week3/day15/ ... day21/  ← Robotics & basic agents
│   ├── week4/day22/ ... day28/  ← NLP track  [projects/ with 3 files/day]
│   ├── week5/day29/ ... day35/  ← Advanced DL [projects/ with 3 files/day]
│   └── week6/day36/ ... day42/  ← Agentic AI  [projects/ with 3 files/day]
│       Each day: README.md, lesson.py, exercises.py, solutions.py
│                 projects/project1_*.py, project2_*.py, project3_*.py
├── ML/          ← ML pipeline templates
├── DL/          ← DL training loop templates + AE/GAN/Transformer
├── CNN/         ← CNN architecture cheatsheet
├── NLP/         ← NLP pipeline templates (CPU + offline fallbacks)
├── ROBOTICS/    ← PID + A*/BFS templates
├── AGENTIC_AI/  ← Agent templates: reflex, planning, multi-agent
├── requirements/
│   ├── beginner-requirements.txt
│   └── advanced-requirements.txt
├── SETUP.md
└── README.md                    ← this file
```

---

## Hardware Notes

All scripts are tuned for **8 GB RAM, no GPU**:

- PyTorch: CPU-only wheel (`torch 2.12.1+cpu`)
- TensorFlow: `tensorflow-cpu 2.21.0`
- Datasets: MNIST subsets <= 5000, synthetic data, sklearn built-ins
- Models: 2-4 conv layers, `batch_size=16-32`, `epochs=3-10`
- Lessons marked `# NOTE: heavier lesson` — close other apps first

---

## Learning Path Recommendations

**Complete Python beginner** — Start at `BEGINNER/day01`

**Knows Python basics** — Start at `ADVANCED/week1/day01`

**Knows data science** — Start at `ADVANCED/week2/day08` (deep learning)

**Wants robotics/AI** — Start at `ADVANCED/week3/day15`

**Wants NLP** — Start at `ADVANCED/week4/day22` (CPU-only, offline fallbacks)

**Wants advanced DL** — Start at `ADVANCED/week5/day31` (autoencoders, GANs, transformers)

**Wants Agentic AI** — Start at `ADVANCED/week6/day37` (planning, tool use, multi-agent)

**Just need templates** — Go directly to `ML/`, `DL/`, `CNN/`, `NLP/`, `ROBOTICS/`, `AGENTIC_AI/`

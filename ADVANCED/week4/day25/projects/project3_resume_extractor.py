"""
Project: Resume Information Extractor
Teaches: practical NER on structured documents — extracting name, email, phone,
         skills, years of experience from a résumé using regex patterns.
~15 MB RAM, ~1s on CPU
"""
import re
from collections import defaultdict

RESUME = """
John Michael Smith
Software Engineer | AI Specialist

Contact:
  Email: john.smith@email.com
  Phone: +1-555-234-5678
  LinkedIn: linkedin.com/in/johnsmith
  GitHub: github.com/jmsmith

Summary:
  Experienced software engineer with 7 years of experience building scalable
  ML systems. Worked at Google, Stripe, and DeepMind.

Education:
  BSc Computer Science, MIT, Cambridge, Massachusetts, 2017
  MSc Artificial Intelligence, Stanford University, 2019

Experience:
  Senior ML Engineer, Google, Mountain View, CA (Jan 2021 – Present)
    - Built recommendation system serving 500M users
    - Led team of 8 engineers in San Francisco

  ML Engineer, Stripe, San Francisco, CA (Jun 2019 – Dec 2020)
    - Designed fraud detection pipeline reducing fraud by 40%

  Research Intern, DeepMind, London, UK (May 2018 – Aug 2018)

Skills:
  Programming: Python, Java, C++, Go, SQL, Bash
  ML/AI: PyTorch, TensorFlow, Scikit-learn, Hugging Face, LangChain
  Cloud: AWS, GCP, Docker, Kubernetes
  Tools: Git, Linux, Spark, Kafka

Certifications:
  AWS Certified Solutions Architect, 2022
  Google Professional ML Engineer, 2023
"""

# ─── Extractors ───────────────────────────────────────────────────────────────
def extract_email(text):
    return re.findall(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+", text)

def extract_phone(text):
    return re.findall(r"\+?1?[-.\s]?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}", text)

def extract_years_exp(text):
    matches = re.findall(r"(\d+)\s+years?\s+of\s+experience", text, re.IGNORECASE)
    return [int(m) for m in matches]

def extract_orgs(text):
    ORG_PATTERNS = r"\b(?:Google|Stripe|DeepMind|Microsoft|Apple|Amazon|Meta|OpenAI|MIT|Stanford)\b"
    return list(set(re.findall(ORG_PATTERNS, text)))

def extract_skills(text):
    SKILL_PATTERNS = r"\b(?:Python|Java|C\+\+|Go|SQL|Bash|PyTorch|TensorFlow|Scikit-learn|"
    SKILL_PATTERNS += r"Hugging Face|LangChain|AWS|GCP|Docker|Kubernetes|Git|Linux|Spark|Kafka)\b"
    return list(set(re.findall(SKILL_PATTERNS, text)))

def extract_dates(text):
    patterns = [
        r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{4}\b",
        r"\b\d{4}\b",
    ]
    dates = []
    for p in patterns:
        dates.extend(re.findall(p, text))
    return sorted(set(dates))

def extract_gpe(text):
    GPE = r"\b(?:California|Massachusetts|London|San Francisco|Mountain View|Cambridge|"
    GPE += r"New York|Chicago|Boston|Seattle|Austin|UK|USA)\b"
    return list(set(re.findall(GPE, text)))

def extract_name(text):
    lines = [l.strip() for l in text.strip().splitlines() if l.strip()]
    for line in lines[:3]:
        if re.match(r"^[A-Z][a-z]+(?: [A-Z][a-z]+){1,3}$", line):
            return line
    return "Unknown"

# ─── Run extraction ───────────────────────────────────────────────────────────
extracted = {
    "Name":          extract_name(RESUME),
    "Email":         extract_email(RESUME),
    "Phone":         extract_phone(RESUME),
    "Years Exp":     extract_years_exp(RESUME),
    "Organizations": extract_orgs(RESUME),
    "Skills":        sorted(extract_skills(RESUME)),
    "Dates":         extract_dates(RESUME),
    "Locations":     extract_gpe(RESUME),
}

print("=== Resume Information Extractor ===\n")
for field, value in extracted.items():
    if isinstance(value, list):
        print(f"  {field:<16}: {', '.join(str(v) for v in value)}")
    else:
        print(f"  {field:<16}: {value}")

# ─── Skill category summary ───────────────────────────────────────────────────
SKILL_CATS = {
    "Languages": ["Python","Java","C++","Go","SQL","Bash"],
    "ML/AI":     ["PyTorch","TensorFlow","Scikit-learn","Hugging Face","LangChain"],
    "Cloud/Ops": ["AWS","GCP","Docker","Kubernetes","Linux","Spark","Kafka","Git"],
}
print("\n  Skill Categories:")
for cat, cat_skills in SKILL_CATS.items():
    found = [s for s in cat_skills if s in extracted["Skills"]]
    print(f"    {cat:<12}: {found}")

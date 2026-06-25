"""
Project: PII Detection & Anonymization Pipeline
Teaches: multi-pattern PII detection, entity-level redaction,
         pseudonymization (consistent replacement), audit trail.
~10 MB RAM, <1s on CPU
"""
import re, hashlib
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

PII_REGISTRY = {
    "SSN":          r"\b\d{3}-\d{2}-\d{4}\b",
    "CREDIT_CARD":  r"\b\d{4}[\s-]\d{4}[\s-]\d{4}[\s-]\d{4}\b",
    "EMAIL":        r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+",
    "PHONE":        r"\b(\+?1[\s-]?)?\(?\d{3}\)?[\s.-]\d{3}[\s.-]\d{4}\b",
    "IP":           r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b",
    "DATE_OF_BIRTH":r"\b(0[1-9]|1[0-2])/(0[1-9]|[12]\d|3[01])/\d{4}\b",
    "PASSPORT":     r"\b[A-Z]{2}\d{7}\b",
}

@dataclass
class PIIEntity:
    pii_type: str
    value:    str
    start:    int
    end:      int
    pseudo:   str = ""

@dataclass
class PIIResult:
    original:    str
    redacted:    str
    pseudonymized: str
    entities:    List[PIIEntity] = field(default_factory=list)

class PIIFilter:
    def __init__(self, pseudonymize=True):
        self.pseudonymize= pseudonymize
        self._pseudo_map: Dict[str, str] = {}  # original → consistent pseudonym
        self.audit: List[PIIResult] = []

    def _pseudo(self, pii_type, value) -> str:
        key = f"{pii_type}:{value}"
        if key not in self._pseudo_map:
            h = hashlib.md5(key.encode()).hexdigest()[:6].upper()
            self._pseudo_map[key] = f"[{pii_type}-{h}]"
        return self._pseudo_map[key]

    def detect(self, text: str) -> List[PIIEntity]:
        entities = []
        for pii_type, pattern in PII_REGISTRY.items():
            for m in re.finditer(pattern, text):
                pseudo = self._pseudo(pii_type, m.group()) if self.pseudonymize else ""
                entities.append(PIIEntity(pii_type, m.group(), m.start(), m.end(), pseudo))
        entities.sort(key=lambda e: e.start)
        return entities

    def process(self, text: str) -> PIIResult:
        entities = self.detect(text)
        redacted = text; pseudonymized = text
        for ent in sorted(entities, key=lambda e: e.start, reverse=True):
            redacted       = redacted[:ent.start]      + f"[REDACTED_{ent.pii_type}]"   + redacted[ent.end:]
            pseudonymized  = pseudonymized[:ent.start] + ent.pseudo                      + pseudonymized[ent.end:]
        result = PIIResult(text, redacted, pseudonymized, entities)
        self.audit.append(result)
        return result

    def report(self):
        total    = sum(len(r.entities) for r in self.audit)
        by_type  = {}
        for r in self.audit:
            for e in r.entities:
                by_type[e.pii_type] = by_type.get(e.pii_type, 0) + 1
        print(f"\n  PII AUDIT: {len(self.audit)} documents, {total} entities detected")
        for t, cnt in sorted(by_type.items(), key=lambda x: -x[1]):
            print(f"    {t:<15}: {cnt}")

pf = PIIFilter(pseudonymize=True)
DOCUMENTS = [
    "Customer John Smith, SSN 123-45-6789, email john@example.com, phone 555-123-4567.",
    "Passport AB1234567. DOB 01/15/1985. Credit card 4532 1234 5678 9012.",
    "Server at 192.168.1.100 received request from 10.0.0.42.",
    "Please call me at (800) 555-0199 or email support@company.org.",
    "No PII in this document — just regular business content.",
]

print("=== PII Detection & Anonymization Pipeline ===\n")
for i, doc in enumerate(DOCUMENTS, 1):
    result = pf.process(doc)
    print(f"  Document {i}:")
    print(f"  Original     : {result.original[:80]}")
    print(f"  Redacted     : {result.redacted[:80]}")
    print(f"  Pseudonymized: {result.pseudonymized[:80]}")
    if result.entities:
        print(f"  Entities     : {[(e.pii_type, e.value[:20]) for e in result.entities]}")
    print()

# Test pseudonymization consistency (same SSN → same pseudonym)
print("  Consistency check — same SSN in two documents:")
doc_a = pf.process("First doc: SSN 123-45-6789")
doc_b = pf.process("Second doc: SSN 123-45-6789")
ent_a = doc_a.entities[0].pseudo; ent_b = doc_b.entities[0].pseudo
print(f"    Doc A pseudo: {ent_a}")
print(f"    Doc B pseudo: {ent_b}")
print(f"    Consistent  : {ent_a == ent_b}")
pf.report()

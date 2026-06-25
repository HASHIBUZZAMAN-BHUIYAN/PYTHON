"""
Project: Content Creation Pipeline
Teaches: multi-stage content pipeline (research→outline→write→review→publish),
         inter-agent message passing, quality gates.
~10 MB RAM, <1s on CPU
"""
from dataclasses import dataclass, field
from typing import List, Dict

@dataclass
class Content:
    topic:     str
    outline:   List[str] = field(default_factory=list)
    sections:  Dict[str, str] = field(default_factory=dict)
    feedback:  List[str] = field(default_factory=list)
    published: bool = False
    quality:   float = 0.0

class PipelineAgent:
    def __init__(self, name, role):
        self.name = name; self.role = role
    def process(self, content: Content) -> Content: raise NotImplementedError

class ResearchAgent(PipelineAgent):
    def __init__(self): super().__init__("Alex","Researcher")
    def process(self, c):
        print(f"  [{self.name}] Researching '{c.topic}'...")
        c.outline = [f"1. Introduction to {c.topic}",
                     f"2. Historical background of {c.topic}",
                     f"3. Key concepts and mechanisms",
                     f"4. Real-world applications",
                     f"5. Future outlook and conclusion"]
        return c

class OutlineAgent(PipelineAgent):
    def __init__(self): super().__init__("Beth","Outliner")
    def process(self, c):
        print(f"  [{self.name}] Expanding outline ({len(c.outline)} sections)...")
        for section in c.outline:
            num  = section.split(".")[0]
            name = section.split(".",1)[1].strip()
            c.sections[section] = f"[DRAFT] Section {num} on {name}: {c.topic} involves..."
        return c

class WriterAgent(PipelineAgent):
    def __init__(self): super().__init__("Carlos","Writer")
    def process(self, c):
        print(f"  [{self.name}] Writing {len(c.sections)} sections...")
        for key in c.sections:
            stub = c.sections[key]
            c.sections[key] = stub.replace("[DRAFT]", "[WRITTEN]") + \
                f" This section covers the essential aspects of {c.topic}."
        return c

class ReviewAgent(PipelineAgent):
    def __init__(self, quality_threshold=0.6): super().__init__("Dana","Reviewer"); self.threshold=quality_threshold
    def process(self, c):
        print(f"  [{self.name}] Reviewing content...")
        word_count = sum(len(s.split()) for s in c.sections.values())
        c.quality  = min(1.0, word_count / 200)
        c.feedback.append(f"Word count: {word_count}. Quality score: {c.quality:.2f}")
        if c.quality >= self.threshold:
            c.feedback.append("APPROVED: Content meets quality standards.")
        else:
            c.feedback.append("REJECTED: Content needs more detail.")
        return c

class PublishAgent(PipelineAgent):
    def __init__(self): super().__init__("Eva","Publisher")
    def process(self, c):
        approved = any("APPROVED" in f for f in c.feedback)
        if approved:
            c.published = True
            print(f"  [{self.name}] PUBLISHED: '{c.topic}'")
        else:
            print(f"  [{self.name}] NOT PUBLISHED — quality check failed")
        return c

class ContentPipeline:
    def __init__(self, agents, max_retries=2):
        self.agents = agents; self.max_retries = max_retries

    def run(self, topic):
        content = Content(topic=topic)
        print(f"\nPipeline starting for: '{topic}'")
        for agent in self.agents:
            for attempt in range(self.max_retries+1):
                try:
                    content = agent.process(content)
                    break
                except Exception as e:
                    print(f"  ERROR in {agent.name}: {e}  (attempt {attempt+1})")
        return content

print("=== Content Creation Pipeline ===")
pipeline = ContentPipeline([ResearchAgent(),OutlineAgent(),WriterAgent(),ReviewAgent(),PublishAgent()])
for topic in ["Machine Learning", "Blockchain Technology"]:
    result = pipeline.run(topic)
    print(f"  Quality: {result.quality:.2f}  Published: {result.published}")
    for fb in result.feedback: print(f"    → {fb}")
    print()

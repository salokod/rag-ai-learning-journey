import chromadb
import re

client = chromadb.Client()

long_report = """COMPREHENSIVE DRAFT SCOUTING REPORT: QUARTERBACK EVALUATION

1. ARM TALENT
Strong arm with measured velocity of 62 mph at the combine. Can make every NFL throw including the deep out and the skinny post from the far hash. Shows good touch on deep balls, placing them over the receiver's outside shoulder. Velocity on short and intermediate throws is elite, with tight spirals even under duress.

2. ACCURACY AND BALL PLACEMENT
68% completion rate over three college seasons. Best in class on intermediate routes (15-25 yards) where he completes 74% of attempts. Anticipation throws to the sideline are a strength. Accuracy declines on deep shots beyond 30 yards, completing only 41% of attempts. Ball placement on back-shoulder fades needs refinement.

3. POCKET PRESENCE
Comfortable in a collapsing pocket and navigates pressure with subtle movements. Climbs the pocket naturally when edge pressure closes in. Average time to throw: 2.3 seconds, among the fastest in this draft class. Under clean protection, completes 78% of passes. Under pressure, completion rate drops to 51%.

4. DECISION MAKING AND PROCESSING
Reads the full field on play-action concepts but tends to lock onto his first read on quick-game passes. Interception-worthy play rate: 2.4%, slightly above average. Excels in the RPO game with correct read rate of 89%. Needs to improve reading Cover-2 rotations post-snap.

5. MOBILITY AND ATHLETICISM
4.72 40-yard dash at the combine. Not a dynamic runner but extends plays when the pocket breaks down. Rushed for 342 yards and 5 touchdowns last season. Runs a controlled scramble style. Slides well and protects himself in the open field.

6. LEADERSHIP AND INTANGIBLES
Team captain for two consecutive seasons. First to arrive, last to leave per coaches. Film study habits graded as elite by coaching staff. Commands the huddle with confidence. 4-1 record in games decided by one score or less."""


# Strategy 1: Fixed-size chunks (500 chars with overlap)
def chunk_fixed(text, size=500, overlap=100):
    chunks = []
    start = 0
    while start < len(text):
        end = start + size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = end - overlap
    return chunks

# Strategy 2: Section-based (what we just did)
def chunk_sections(text):
    sections = re.split(r'\n(?=\d+\.)', text.strip())
    return [s.strip() for s in sections if s.strip()]

# Strategy 3: Paragraph-based
def chunk_paragraphs(text):
    paras = text.strip().split('\n\n')
    return [p.strip() for p in paras if p.strip()]


fixed = chunk_fixed(long_report)
sections = chunk_sections(long_report)
paragraphs = chunk_paragraphs(long_report)

print(f"Fixed-size:  {len(fixed)} chunks")
for i, c in enumerate(fixed):
    print(f"  {i}: {len(c)} chars -- '{c[:50]}...'")

print(f"\nSection:     {len(sections)} chunks")
for i, c in enumerate(sections):
    print(f"  {i}: {len(c)} chars -- '{c.split(chr(10))[0]}'")

print(f"\nParagraph:   {len(paragraphs)} chunks")
for i, c in enumerate(paragraphs):
    print(f"  {i}: {len(c)} chars -- '{c[:50]}...'")

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

# --- Approach 1: whole document as a single chunk ---
whole_doc = client.create_collection(name="whole_doc")
whole_doc.add(ids=["full"], documents=[long_report])

results = whole_doc.query(query_texts=["What is his completion rate under pressure?"], n_results=1)
print("=== Whole doc result ===")
# Returns the entire report -- the answer is in there, buried with 5 other sections
print(results["documents"][0][0][:120], "...")
print(f"Distance: {results['distances'][0][0]:.3f}")

# --- Approach 2: split by numbered section ---
sections = re.split(r'\n(?=\d+\.)', long_report.strip())
sections = [s.strip() for s in sections if s.strip()]

print(f"\nSplit into {len(sections)} sections:")
for i, s in enumerate(sections):
    first_line = s.split('\n')[0]
    print(f"  Section {i}: {first_line} ({len(s)} chars)")

chunked_doc = client.create_collection(name="chunked_doc")
for i, section in enumerate(sections):
    chunked_doc.add(
        ids=[f"section-{i}"],
        documents=[section],
        metadatas=[{"section_index": i, "source": "QB-EVAL-001"}],
    )

results = chunked_doc.query(query_texts=["What is his completion rate under pressure?"], n_results=1)
print("\n=== Chunked result ===")
# Returns just the POCKET PRESENCE section -- precise, lower distance
print(results["documents"][0][0])
print(f"Distance: {results['distances'][0][0]:.3f}")

# --- Compare both approaches across several queries ---
print("\n=== Comparison: whole doc vs chunked ===")
test_queries = [
    "How does he handle pre-snap reads?",
    "What are his leadership qualities?",
    "How fast is he in the 40-yard dash?",
]

for query in test_queries:
    r_whole = whole_doc.query(query_texts=[query], n_results=1)
    r_chunk = chunked_doc.query(query_texts=[query], n_results=1)

    # Show which section the chunked approach found
    chunk_section = r_chunk["documents"][0][0].split('\n')[0]
    print(f"\nQuery: '{query}'")
    print(f"  Whole doc distance: {r_whole['distances'][0][0]:.3f}")
    print(f"  Chunk distance:     {r_chunk['distances'][0][0]:.3f} --> {chunk_section}")

import chromadb

client = chromadb.Client()

collection = client.create_collection(name="scouting_reports")

collection.add(
    ids=["QB-101"],
    documents=["Pocket passer with elite accuracy. Completes 68% of passes with a 2.3-second average release time. Excels on intermediate routes (15-25 yards) with anticipation throws. Reads defenses pre-snap and adjusts protection. Arm strength measured at 62 mph at the combine. Weakness: locks onto first read under heavy pressure."],
    metadatas=[{"position": "QB", "report_type": "scouting"}],
)

collection.add(
    ids=["RB-201"],
    documents=["Explosive runner with 4.38 40-yard dash. Exceptional vision through traffic and finds cutback lanes consistently. Averages 3.8 yards after contact per carry. Reliable pass catcher out of the backfield with 45 receptions last season. Weakness: needs to improve pass protection and blitz pickup."],
    metadatas=[{"position": "RB", "report_type": "scouting"}],
)

collection.add(
    ids=["WR-301"],
    documents=["Crisp route runner with elite separation at the top of routes. Runs the full route tree from slot and outside. 4.42 speed with a 38-inch vertical leap. Reliable hands with a 2.1% drop rate. Weakness: struggles against physical press coverage at the line of scrimmage."],
    metadatas=[{"position": "WR", "report_type": "scouting"}],
)

collection.add(
    ids=["OL-401"],
    documents=["Excellent anchor in pass protection with quick lateral movement to mirror speed rushers. 34-inch arm length provides leverage advantage. Run blocking grade: 82.5 out of 100. Allowed only 2 sacks in 580 pass-blocking snaps last season. Weakness: combo blocks to the second level."],
    metadatas=[{"position": "OL", "report_type": "scouting"}],
)

collection.add(
    ids=["DEF-501"],
    documents=["Cover-3 base defense with single-high safety. Corners play press technique on early downs. Linebackers run pattern-match zone on 3rd and long. Aggressive blitz from nickel and dime personnel. Tendency: susceptible to crossing routes against zone coverage."],
    metadatas=[{"position": "DEF", "report_type": "scheme"}],
)

collection.add(
    ids=["SPEC-601"],
    documents=["Punter averages 46.2 yards per punt with 4.1-second hang time. Directional kicking grade: elite. Coffin corner accuracy: 73% inside the 10-yard line. Kickoff specialist reaches the end zone on 88% of attempts. Coverage units allow 7.2 average return yards."],
    metadatas=[{"position": "SPEC", "report_type": "scouting"}],
)

print(f"Total docs: {collection.count()}")

# Only search scouting reports
results = collection.query(
    query_texts=["who has the best measurables"],
    n_results=3,
    where={"report_type": "scouting"},
)

print("Scouting reports matching 'who has the best measurables':")
for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
    print(f"  [{meta['position']}] {doc[:70]}...")

result2 = collection.query(
    query_texts=["pass coverage tendencies"],
    n_results=3,
    where={"report_type": "scheme"},
)

print("\nScheme reports matching 'pass coverage tendencies':")
for doc, meta in zip(result2["documents"][0], result2["metadatas"][0]):
    print(f"  [{meta['position']}] {doc[:70]}...")

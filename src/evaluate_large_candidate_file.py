import pandas as pd
import sys
from pathlib import Path

if len(sys.argv) != 2:
    print("Usage:")
    print("python src\\evaluate_large_candidate_file.py <candidate_file>")
    sys.exit(1)

candidate_file = Path(sys.argv[1])

GT_FILE = Path("experiments/001_baseline/validation_ground_truth.tsv")

CHUNK_SIZE = 500_000

print("=" * 80)
print("LARGE CANDIDATE BLOCKING EVALUATION")
print("=" * 80)

print(f"\nCandidate file: {candidate_file}")
print(f"Ground truth:   {GT_FILE}")

# =========================================================
# LOAD GROUND TRUTH
# =========================================================

print("\nLoading validation ground truth...")

gt = pd.read_csv(
    GT_FILE,
    sep="\t",
    dtype=str
)

gt["matched_entity_ids"] = gt["matched_entity_ids"].fillna("")

print(f"Validation S1 records: {len(gt):,}")

# =========================================================
# BUILD TRUE PAIRS
# =========================================================

print("\nBuilding true-pair lookup...")

true_pairs = set()

for row in gt.itertuples(index=False):
    s1 = row.source1_entity_id
    matches = row.matched_entity_ids

    if matches:
        for entity_id in matches.split(","):
            true_pairs.add((s1, entity_id))

print(f"True match pairs: {len(true_pairs):,}")

# True S1 entities
true_s1 = set()

for s1, _ in true_pairs:
    true_s1.add(s1)

print(f"S1 entities with >=1 true match: {len(true_s1):,}")

# =========================================================
# READ CANDIDATE FILE IN CHUNKS
# =========================================================

print("\nReading candidate file in chunks...")

retrieved_pairs = set()

total_rows = 0
chunk_number = 0

for chunk in pd.read_csv(
    candidate_file,
    sep="\t",
    dtype=str,
    usecols=["source1_entity_id", "candidate_entity_id"],
    chunksize=CHUNK_SIZE
):

    chunk_number += 1
    total_rows += len(chunk)

    print(
        f"\rProcessed rows: {total_rows:,}",
        end="",
        flush=True
    )

    # Convert only this chunk into tuples
    chunk_pairs = zip(
        chunk["source1_entity_id"],
        chunk["candidate_entity_id"]
    )

    # Keep only true pairs found in this chunk
    for pair in chunk_pairs:
        if pair in true_pairs:
            retrieved_pairs.add(pair)

print("\n")

# =========================================================
# RESULTS
# =========================================================

retrieved = len(retrieved_pairs)

missed = len(true_pairs) - retrieved

recall = retrieved / len(true_pairs)

print("=" * 80)
print("RESULT")
print("=" * 80)

print(f"\nCandidate rows:        {total_rows:,}")
print(f"True match pairs:      {len(true_pairs):,}")
print(f"Retrieved true pairs:  {retrieved:,}")
print(f"Missed true pairs:     {missed:,}")
print(f"Pair-level recall:     {recall:.4%}")

# =========================================================
# S1 COVERAGE
# =========================================================

retrieved_s1 = {
    s1
    for s1, _ in retrieved_pairs
}

covered = len(retrieved_s1)

coverage = covered / len(true_s1)

print(f"\nS1 with >=1 true match:       {len(true_s1):,}")
print(f"S1 with >=1 retrieved match:  {covered:,}")
print(f"S1 coverage:                  {coverage:.4%}")

# =========================================================
# BASIC CANDIDATE STATS
# =========================================================

print("\n" + "=" * 80)
print("CANDIDATE STATISTICS")
print("=" * 80)

print(f"\nCandidate rows:        {total_rows:,}")

print("\nEvaluation complete.")
import pandas as pd
import sys
from pathlib import Path

if len(sys.argv) != 2:
    print("Usage:")
    print("python src\\evaluate_candidate_file.py <candidate_file>")
    sys.exit(1)

candidate_file = Path(sys.argv[1])

GT_FILE = Path("experiments/001_baseline/validation_ground_truth.tsv")

print("=" * 80)
print("CANDIDATE BLOCKING EVALUATION")
print("=" * 80)

print(f"\nCandidate file: {candidate_file}")
print(f"Ground truth:   {GT_FILE}")

# ---------------------------------------------------------
# Load ground truth
# ---------------------------------------------------------

print("\nLoading validation ground truth...")

gt = pd.read_csv(
    GT_FILE,
    sep="\t",
    dtype=str
)

gt["matched_entity_ids"] = gt["matched_entity_ids"].fillna("")

print(f"Validation S1 records: {len(gt):,}")

# ---------------------------------------------------------
# Build true pair set
# ---------------------------------------------------------

true_pairs = set()

for _, row in gt.iterrows():
    s1 = row["source1_entity_id"]
    matches = row["matched_entity_ids"]

    if matches:
        for s2s3 in matches.split(","):
            true_pairs.add((s1, s2s3))

print(f"True match pairs: {len(true_pairs):,}")

# ---------------------------------------------------------
# Load candidates
# ---------------------------------------------------------

print("\nLoading candidate file...")

candidates = pd.read_csv(
    candidate_file,
    sep="\t",
    dtype=str
)

print(f"Candidate rows: {len(candidates):,}")

# ---------------------------------------------------------
# Candidate pair set
# ---------------------------------------------------------

candidate_pairs = set(
    zip(
        candidates["source1_entity_id"],
        candidates["candidate_entity_id"]
    )
)

print(f"Unique candidate pairs: {len(candidate_pairs):,}")

# ---------------------------------------------------------
# Pair-level recall
# ---------------------------------------------------------

retrieved = len(true_pairs & candidate_pairs)

missed = len(true_pairs - candidate_pairs)

recall = retrieved / len(true_pairs)

print("\n" + "=" * 80)
print("RESULT")
print("=" * 80)

print(f"\nTrue match pairs:       {len(true_pairs):,}")
print(f"Retrieved true pairs:  {retrieved:,}")
print(f"Missed true pairs:     {missed:,}")
print(f"Pair-level recall:     {recall:.4%}")

# ---------------------------------------------------------
# S1 coverage
# ---------------------------------------------------------

true_by_s1 = {}

for s1, entity in true_pairs:
    true_by_s1.setdefault(s1, set()).add(entity)

candidate_by_s1 = {}

for s1, entity in candidate_pairs:
    candidate_by_s1.setdefault(s1, set()).add(entity)

covered = 0

for s1 in true_by_s1:
    if true_by_s1[s1] & candidate_by_s1.get(s1, set()):
        covered += 1

coverage = covered / len(true_by_s1)

print(f"\nS1 with >=1 true match:       {len(true_by_s1):,}")
print(f"S1 with >=1 retrieved match:  {covered:,}")
print(f"S1 coverage:                  {coverage:.4%}")

# ---------------------------------------------------------
# Candidate statistics
# ---------------------------------------------------------

print("\n" + "=" * 80)
print("CANDIDATE STATISTICS")
print("=" * 80)

print(f"\nCandidate rows:           {len(candidates):,}")
print(f"Unique candidate pairs:   {len(candidate_pairs):,}")

print("\nEvaluation complete.")
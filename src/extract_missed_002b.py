from pathlib import Path
import pandas as pd


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------
BASE = Path("experiments/001_baseline")

GROUND_TRUTH = BASE / "validation_ground_truth.tsv"
CANDIDATES = BASE / "approx_candidates.tsv"
OUTPUT = BASE / "missed_002b_pairs.tsv"

CHUNK_SIZE = 500_000


# ---------------------------------------------------------
# Load true match pairs
# ---------------------------------------------------------
print("Loading validation ground truth...")

gt = pd.read_csv(
    GROUND_TRUTH,
    sep="\t",
    dtype=str
)

true_pairs = set()

for _, row in gt.iterrows():
    s1 = row["source1_entity_id"]
    matched = row["matched_entity_ids"]

    if pd.isna(matched) or str(matched).strip() == "":
        continue

    for candidate_id in str(matched).split(","):
        candidate_id = candidate_id.strip()

        if candidate_id:
            true_pairs.add((s1, candidate_id))

print(f"True match pairs: {len(true_pairs):,}")


# ---------------------------------------------------------
# Stream candidate file
# ---------------------------------------------------------
print("\nScanning candidate file...")

remaining = true_pairs.copy()

processed = 0
chunk_number = 0

with open(OUTPUT, "w", encoding="utf-8", newline="") as f:

    f.write("source1_entity_id\tcandidate_entity_id\n")

    for chunk in pd.read_csv(
        CANDIDATES,
        sep="\t",
        dtype=str,
        chunksize=CHUNK_SIZE
    ):

        chunk_number += 1
        processed += len(chunk)

        for s1, candidate_id in zip(
            chunk["source1_entity_id"],
            chunk["candidate_entity_id"]
        ):
            pair = (s1, candidate_id)

            if pair in remaining:
                remaining.remove(pair)

        if chunk_number % 10 == 0:
            found = len(true_pairs) - len(remaining)

            print(
                f"Processed: {processed:,} | "
                f"Retrieved: {found:,} | "
                f"Remaining missed: {len(remaining):,}"
            )


# ---------------------------------------------------------
# Write missed true pairs
# ---------------------------------------------------------
print("\nWriting missed true pairs...")

with open(OUTPUT, "a", encoding="utf-8", newline="") as f:

    for s1, candidate_id in sorted(remaining):
        f.write(f"{s1}\t{candidate_id}\n")


# ---------------------------------------------------------
# Final summary
# ---------------------------------------------------------
retrieved = len(true_pairs) - len(remaining)

print("\n========================================")
print("002B MISSED-PAIR EXTRACTION COMPLETE")
print("========================================")
print(f"True pairs:       {len(true_pairs):,}")
print(f"Retrieved pairs:  {retrieved:,}")
print(f"Missed pairs:     {len(remaining):,}")
print(f"Output:           {OUTPUT}")
print("========================================")
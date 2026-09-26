from pathlib import Path
import random
import pandas as pd


# ============================================================
# Configuration
# ============================================================

GT_FILE = Path("experiments/001_baseline/train_ground_truth.tsv")

OUTPUT_DIR = Path("experiments/005_ml_training")
OUTPUT_FILE = OUTPUT_DIR / "training_pairs.tsv"

TARGET_POSITIVES = 1_000_000
NEGATIVE_RATIO = 1

RANDOM_STATE = 42

S2_FILE = Path("dataset/train/train_source2.tsv")
S3_FILE = Path("dataset/train/train_source3.tsv")


# ============================================================
# Load development ground truth
# ============================================================

print("=" * 70)
print("B-06B: Create ML Training Pairs")
print("=" * 70)

print("\nLoading development ground truth...")

gt = pd.read_csv(
    GT_FILE,
    sep="\t",
    keep_default_na=False,
)

print(f"S1 rows: {len(gt):,}")


# ============================================================
# Parse positive pairs
# ============================================================

print("\nParsing positive pairs...")

positive_rows = []

for row in gt.itertuples(index=False):
    s1_id = row.source1_entity_id
    matched = row.matched_entity_ids

    if not matched:
        continue

    for candidate_id in matched.split(","):
        if candidate_id:
            positive_rows.append(
                (s1_id, candidate_id, 1)
            )

print(f"Total positive pairs available: {len(positive_rows):,}")


# ============================================================
# Sample positives
# ============================================================

rng = random.Random(RANDOM_STATE)

if TARGET_POSITIVES < len(positive_rows):
    print(
        f"Sampling {TARGET_POSITIVES:,} positive pairs "
        f"from {len(positive_rows):,}"
    )

    positive_rows = rng.sample(
        positive_rows,
        TARGET_POSITIVES,
    )
else:
    print("Using all available positive pairs.")


positive_df = pd.DataFrame(
    positive_rows,
    columns=[
        "source1_entity_id",
        "candidate_entity_id",
        "label",
    ],
)

print(f"Selected positives: {len(positive_df):,}")


# ============================================================
# Build set of known positives
# ============================================================

print("\nBuilding positive-pair lookup...")

positive_keys = set(
    zip(
        positive_df["source1_entity_id"],
        positive_df["candidate_entity_id"],
    )
)

print(f"Positive lookup size: {len(positive_keys):,}")


# ============================================================
# Load S2/S3 IDs
# ============================================================

print("\nLoading candidate entity IDs...")

s2_ids = pd.read_csv(
    S2_FILE,
    sep="\t",
    usecols=["entity_id"],
)["entity_id"].tolist()

print(f"S2 IDs: {len(s2_ids):,}")

s3_ids = pd.read_csv(
    S3_FILE,
    sep="\t",
    usecols=["entity_id"],
)["entity_id"].tolist()

print(f"S3 IDs: {len(s3_ids):,}")


candidate_ids = s2_ids + s3_ids

print(f"Total candidate IDs: {len(candidate_ids):,}")


# ============================================================
# Negative sampling
# ============================================================

target_negatives = len(positive_df) * NEGATIVE_RATIO

print(
    f"\nGenerating {target_negatives:,} negative pairs..."
)

# We generate negatives by selecting an S1 and a random
# S2/S3 candidate, rejecting known positive pairs.

s1_ids = gt["source1_entity_id"].tolist()

negative_pairs = set()

attempts = 0
max_attempts = target_negatives * 20

while len(negative_pairs) < target_negatives:

    attempts += 1

    if attempts > max_attempts:
        raise RuntimeError(
            "Unable to generate the requested number of unique "
            "negative pairs within the attempt limit."
        )

    s1_id = rng.choice(s1_ids)
    candidate_id = rng.choice(candidate_ids)

    key = (s1_id, candidate_id)

    if key in positive_keys:
        continue

    negative_pairs.add(key)

    if len(negative_pairs) % 100_000 == 0:
        print(
            f"  negatives generated: "
            f"{len(negative_pairs):,}"
        )


negative_df = pd.DataFrame(
    [
        (s1_id, candidate_id, 0)
        for s1_id, candidate_id in negative_pairs
    ],
    columns=[
        "source1_entity_id",
        "candidate_entity_id",
        "label",
    ],
)

print(
    f"Selected negatives: {len(negative_df):,}"
)


# ============================================================
# Combine
# ============================================================

training_df = pd.concat(
    [
        positive_df,
        negative_df,
    ],
    ignore_index=True,
)


# Shuffle deterministically
training_df = training_df.sample(
    frac=1.0,
    random_state=RANDOM_STATE,
).reset_index(drop=True)


# ============================================================
# Integrity checks
# ============================================================

print("\nRunning integrity checks...")

duplicate_count = training_df.duplicated(
    subset=[
        "source1_entity_id",
        "candidate_entity_id",
    ]
).sum()

if duplicate_count != 0:
    raise RuntimeError(
        f"Duplicate pairs found: {duplicate_count}"
    )

print("Duplicate pairs: 0")


label_counts = training_df["label"].value_counts().sort_index()

print("\nLabel distribution:")
print(label_counts.to_string())


# Check that no negative is a known positive
negative_keys = set(
    zip(
        negative_df["source1_entity_id"],
        negative_df["candidate_entity_id"],
    )
)

overlap = negative_keys.intersection(positive_keys)

if overlap:
    raise RuntimeError(
        f"Negative/positive overlap detected: {len(overlap)}"
    )

print("Negative/positive overlap: 0")


# ============================================================
# Save
# ============================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

training_df.to_csv(
    OUTPUT_FILE,
    sep="\t",
    index=False,
)

print("\nSaved:")
print(OUTPUT_FILE)

print(f"\nTotal training pairs: {len(training_df):,}")

print("\nB-06B complete.")
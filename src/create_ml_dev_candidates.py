from pathlib import Path
import random
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]

TRAIN_DIR = BASE_DIR / "dataset" / "train"
EXP_DIR = BASE_DIR / "experiments" / "001_baseline"
OUTPUT_DIR = BASE_DIR / "experiments" / "004_ml_baseline"

GT_PATH = EXP_DIR / "validation_ground_truth.tsv"
OUTPUT_PATH = OUTPUT_DIR / "dev_candidate_pairs.tsv"

RANDOM_SEED = 42
NEGATIVE_MULTIPLIER = 2


def load_ground_truth():
    gt = pd.read_csv(
        GT_PATH,
        sep="\t",
        dtype=str,
        keep_default_na=False,
    )

    return gt


def build_positive_pairs(gt):
    rows = []

    for row in gt.itertuples(index=False):
        s1_id = row.source1_entity_id
        matched = row.matched_entity_ids

        if not matched:
            continue

        for candidate_id in matched.split(","):
            candidate_id = candidate_id.strip()

            if candidate_id:
                rows.append(
                    {
                        "source1_entity_id": s1_id,
                        "candidate_entity_id": candidate_id,
                        "label": 1,
                    }
                )

    return pd.DataFrame(rows)


def load_source_ids(filename):
    path = TRAIN_DIR / filename

    df = pd.read_csv(
        path,
        sep="\t",
        usecols=["entity_id"],
        dtype=str,
    )

    return df["entity_id"].tolist()


def build_negative_pairs(gt, positive_pairs):
    rng = random.Random(RANDOM_SEED)

    positive_keys = set(
        zip(
            positive_pairs["source1_entity_id"],
            positive_pairs["candidate_entity_id"],
        )
    )

    s1_ids = gt["source1_entity_id"].tolist()

    s2_ids = load_source_ids("train_source2.tsv")
    s3_ids = load_source_ids("train_source3.tsv")

    candidate_ids = s2_ids + s3_ids

    target_negatives = len(positive_pairs) * NEGATIVE_MULTIPLIER

    negatives = []
    seen = set()

    while len(negatives) < target_negatives:
        s1_id = rng.choice(s1_ids)
        candidate_id = rng.choice(candidate_ids)

        key = (s1_id, candidate_id)

        if key in positive_keys or key in seen:
            continue

        seen.add(key)

        negatives.append(
            {
                "source1_entity_id": s1_id,
                "candidate_entity_id": candidate_id,
                "label": 0,
            }
        )

    return pd.DataFrame(negatives)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("MEMBER B - ML DEVELOPMENT CANDIDATE SET")
    print("=" * 80)

    print("\nLoading validation ground truth...")
    gt = load_ground_truth()

    print(f"Validation S1 entities: {len(gt):,}")

    print("\nBuilding positive pairs...")
    positives = build_positive_pairs(gt)

    print(f"Positive pairs: {len(positives):,}")

    print("\nBuilding negative pairs...")
    negatives = build_negative_pairs(gt, positives)

    print(f"Negative pairs: {len(negatives):,}")

    pairs = pd.concat(
        [positives, negatives],
        ignore_index=True,
    )

    pairs = pairs.sample(
        frac=1.0,
        random_state=RANDOM_SEED,
    ).reset_index(drop=True)

    pairs.to_csv(
        OUTPUT_PATH,
        sep="\t",
        index=False,
    )

    print("\n" + "-" * 80)
    print("DEVELOPMENT DATASET CREATED")
    print("-" * 80)

    print(f"Output: {OUTPUT_PATH}")
    print(f"Total pairs: {len(pairs):,}")
    print(f"Positive pairs: {(pairs['label'] == 1).sum():,}")
    print(f"Negative pairs: {(pairs['label'] == 0).sum():,}")

    print("\nLabel distribution:")
    print(pairs["label"].value_counts())

    print("\nFirst 5 rows:")
    print(pairs.head())


if __name__ == "__main__":
    main()
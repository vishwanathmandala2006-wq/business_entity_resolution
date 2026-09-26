from pathlib import Path
from collections import Counter

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent

GT_PATH = (
    BASE_DIR
    / "dataset"
    / "train"
    / "train_ground_truth.tsv"
)


def get_source(entity_id):

    if entity_id.startswith("S2-"):
        return "S2"

    if entity_id.startswith("S3-"):
        return "S3"

    return "OTHER"


def main():

    print("=" * 80)
    print("GROUND TRUTH SOURCE ANALYSIS")
    print("=" * 80)

    df = pd.read_csv(
        GT_PATH,
        sep="\t",
        dtype=str,
        keep_default_na=False
    )

    print("\nTotal S1 entities:", f"{len(df):,}")

    source_pattern_counts = Counter()

    total_s2 = 0
    total_s3 = 0

    s1_zero = 0

    for value in df["matched_entity_ids"]:

        if not value.strip():
            s1_zero += 1
            source_pattern_counts["NONE"] += 1
            continue

        ids = [
            x.strip()
            for x in value.split(",")
            if x.strip()
        ]

        s2_count = sum(
            get_source(x) == "S2"
            for x in ids
        )

        s3_count = sum(
            get_source(x) == "S3"
            for x in ids
        )

        total_s2 += s2_count
        total_s3 += s3_count

        pattern = f"S2={s2_count}, S3={s3_count}"

        source_pattern_counts[pattern] += 1

    print("\nZero-match S1 entities:")
    print(f"  {s1_zero:,}")

    print("\nTotal S2 ground-truth matches:")
    print(f"  {total_s2:,}")

    print("\nTotal S3 ground-truth matches:")
    print(f"  {total_s3:,}")

    print("\nSource composition per S1:")
    print("-" * 50)

    for pattern, count in sorted(
        source_pattern_counts.items(),
        key=lambda x: (
            999 if x[0] == "NONE"
            else int(x[0].split(",")[0].split("=")[1])
            + int(x[0].split(",")[1].split("=")[1])
        )
    ):

        percentage = count / len(df) * 100

        print(
            f"{pattern:15s}"
            f"{count:12,}"
            f"  ({percentage:6.2f}%)"
        )

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
from pathlib import Path
from collections import Counter

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent

EXP_DIR = (
    BASE_DIR
    / "experiments"
    / "001_baseline"
)

GT_PATH = (
    EXP_DIR
    / "validation_ground_truth.tsv"
)

CANDIDATE_PATH = (
    EXP_DIR
    / "baseline_candidates.tsv"
)


def parse_ground_truth(value):

    if not value.strip():
        return []

    return [
        x.strip()
        for x in value.split(",")
        if x.strip()
    ]


def main():

    print("=" * 80)
    print("BASELINE BLOCKING EVALUATION")
    print("=" * 80)

    # -----------------------------------------------------------------
    # Load validation ground truth
    # -----------------------------------------------------------------

    print("\nLoading validation ground truth...")

    gt = pd.read_csv(
        GT_PATH,
        sep="\t",
        dtype=str,
        keep_default_na=False
    )

    print(
        "Validation S1:",
        f"{len(gt):,}"
    )

    # -----------------------------------------------------------------
    # Load candidates
    # -----------------------------------------------------------------

    print("\nLoading candidate pairs...")

    candidates = pd.read_csv(
        CANDIDATE_PATH,
        sep="\t",
        dtype=str,
        keep_default_na=False
    )

    print(
        "Candidate pairs:",
        f"{len(candidates):,}"
    )

    # -----------------------------------------------------------------
    # Create candidate lookup
    # -----------------------------------------------------------------

    candidate_lookup = {}

    for row in candidates.itertuples(index=False):

        key = (
            row.source1_entity_id,
            row.candidate_entity_id
        )

        candidate_lookup[key] = True

    # -----------------------------------------------------------------
    # Evaluate
    # -----------------------------------------------------------------

    total_true_pairs = 0
    retrieved_true_pairs = 0
    missed_true_pairs = 0

    s1_with_true_match = 0
    s1_with_retrieved_match = 0

    missed_examples = []

    miss_reason_counter = Counter()

    for row in gt.itertuples(index=False):

        s1_id = row.source1_entity_id

        true_matches = parse_ground_truth(
            row.matched_entity_ids
        )

        if not true_matches:
            continue

        s1_with_true_match += 1

        retrieved_for_s1 = 0

        for true_id in true_matches:

            total_true_pairs += 1

            key = (
                s1_id,
                true_id
            )

            if key in candidate_lookup:

                retrieved_true_pairs += 1
                retrieved_for_s1 += 1

            else:

                missed_true_pairs += 1

                if len(missed_examples) < 100:

                    missed_examples.append(
                        (
                            s1_id,
                            true_id
                        )
                    )

        if retrieved_for_s1 > 0:
            s1_with_retrieved_match += 1

    # -----------------------------------------------------------------
    # Metrics
    # -----------------------------------------------------------------

    pair_recall = (
        retrieved_true_pairs
        / total_true_pairs
        if total_true_pairs
        else 0
    )

    s1_coverage = (
        s1_with_retrieved_match
        / s1_with_true_match
        if s1_with_true_match
        else 0
    )

    print("\n" + "-" * 80)
    print("RESULTS")
    print("-" * 80)

    print(
        "\nTotal true match pairs:",
        f"{total_true_pairs:,}"
    )

    print(
        "Retrieved true pairs:",
        f"{retrieved_true_pairs:,}"
    )

    print(
        "Missed true pairs:",
        f"{missed_true_pairs:,}"
    )

    print(
        "\nPAIR-LEVEL CANDIDATE RECALL:",
        f"{pair_recall * 100:.4f}%"
    )

    print(
        "\nS1 entities with >=1 true match:",
        f"{s1_with_true_match:,}"
    )

    print(
        "S1 entities with >=1 retrieved match:",
        f"{s1_with_retrieved_match:,}"
    )

    print(
        "\nS1 COVERAGE:",
        f"{s1_coverage * 100:.4f}%"
    )

    # -----------------------------------------------------------------
    # Missed examples
    # -----------------------------------------------------------------

    print("\n" + "-" * 80)
    print("FIRST 100 MISSED TRUE PAIRS")
    print("-" * 80)

    if not missed_examples:

        print("\nNo missed pairs!")

    else:

        for s1_id, true_id in missed_examples:

            print(
                f"{s1_id}\t{true_id}"
            )

    print("\n" + "=" * 80)
    print("BLOCKING EVALUATION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
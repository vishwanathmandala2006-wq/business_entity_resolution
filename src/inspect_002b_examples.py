from pathlib import Path
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_PATH = (
    BASE_DIR
    / "experiments"
    / "001_baseline"
    / "missed_002b_diagnostics.tsv"
)

OUTPUT_PATH = (
    BASE_DIR
    / "experiments"
    / "001_baseline"
    / "002b_representative_examples.tsv"
)


def main():

    df = pd.read_csv(
        INPUT_PATH,
        sep="\t",
        dtype=str
    )

    # Convert numeric columns
    numeric_cols = [
        "name_char_similarity",
        "name_jaccard",
        "address_char_similarity",
        "address_jaccard",
        "address_common_numbers"
    ]

    for col in numeric_cols:
        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )

    groups = []

    # ---------------------------------------------------------
    # Group 1: Strong name, weak address
    # ---------------------------------------------------------

    g1 = df[
        (df["name_char_similarity"] >= 0.85)
        &
        (df["address_char_similarity"] < 0.50)
    ].copy()

    g1 = g1.sample(
        n=min(20, len(g1)),
        random_state=42
    )

    g1["diagnostic_group"] = (
        "strong_name_weak_address"
    )

    groups.append(g1)

    # ---------------------------------------------------------
    # Group 2: Weak name, strong address
    # ---------------------------------------------------------

    g2 = df[
        (df["name_char_similarity"] < 0.70)
        &
        (df["address_char_similarity"] >= 0.85)
    ].copy()

    g2 = g2.sample(
        n=min(20, len(g2)),
        random_state=42
    )

    g2["diagnostic_group"] = (
        "weak_name_strong_address"
    )

    groups.append(g2)

    # ---------------------------------------------------------
    # Group 3: Strong both
    # ---------------------------------------------------------

    g3 = df[
        (df["name_char_similarity"] >= 0.85)
        &
        (df["address_char_similarity"] >= 0.85)
    ].copy()

    g3 = g3.sample(
        n=min(20, len(g3)),
        random_state=42
    )

    g3["diagnostic_group"] = (
        "strong_name_strong_address"
    )

    groups.append(g3)

    # ---------------------------------------------------------
    # Group 4: Weak both
    # ---------------------------------------------------------

    g4 = df[
        (df["name_char_similarity"] < 0.70)
        &
        (df["address_char_similarity"] < 0.70)
    ].copy()

    g4 = g4.sample(
        n=min(20, len(g4)),
        random_state=42
    )

    g4["diagnostic_group"] = (
        "weak_name_weak_address"
    )

    groups.append(g4)

    # ---------------------------------------------------------
    # Group 5: Shared address number
    # ---------------------------------------------------------

    g5 = df[
        (df["address_common_numbers"] >= 1)
        &
        (df["name_char_similarity"] < 0.70)
        &
        (df["address_char_similarity"] < 0.70)
    ].copy()

    g5 = g5.sample(
        n=min(20, len(g5)),
        random_state=42
    )

    g5["diagnostic_group"] = (
        "common_number_weak_text"
    )

    groups.append(g5)

    # ---------------------------------------------------------
    # Combine
    # ---------------------------------------------------------

    result = pd.concat(
        groups,
        ignore_index=True
    )

    # Keep useful columns first
    columns = [
        "diagnostic_group",
        "candidate_source",
        "source1_entity_id",
        "candidate_entity_id",
        "s1_name",
        "candidate_name",
        "s1_address",
        "candidate_address",
        "s1_country",
        "candidate_country",
        "name_char_similarity",
        "name_jaccard",
        "name_char3_jaccard",
        "address_char_similarity",
        "address_jaccard",
        "address_char3_jaccard",
        "address_common_numbers"
    ]

    result = result[
        [c for c in columns if c in result.columns]
    ]

    result.to_csv(
        OUTPUT_PATH,
        sep="\t",
        index=False
    )

    print("=" * 80)
    print("002B REPRESENTATIVE EXAMPLES")
    print("=" * 80)

    print(
        f"\nTotal examples: {len(result):,}"
    )

    print(
        "\nGroup distribution:"
    )

    print(
        result["diagnostic_group"]
        .value_counts()
        .to_string()
    )

    print(
        f"\nSaved to:\n{OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()
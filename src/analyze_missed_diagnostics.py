from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_PATH = (
    BASE_DIR
    / "experiments"
    / "001_baseline"
    / "missed_pair_diagnostics.tsv"
)


def percentage(series):

    return (
        series.mean() * 100
        if len(series) > 0
        else 0
    )


def main():

    print("=" * 80)
    print("ANALYSIS OF MISSED-PAIR DIAGNOSTICS")
    print("=" * 80)

    df = pd.read_csv(
        INPUT_PATH,
        sep="\t",
        dtype=str,
        keep_default_na=False
    )

    numeric_columns = [
        "name_exact_normalized",
        "name_jaccard",
        "name_char_similarity",
        "name_char3_jaccard",
        "address_exact_normalized",
        "address_jaccard",
        "address_char_similarity",
        "address_char3_jaccard",
        "address_common_numbers",
        "country_exact"
    ]

    for col in numeric_columns:
        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        ).fillna(0)

    print(
        "\nDiagnostic pairs:",
        f"{len(df):,}"
    )

    # -------------------------------------------------------------
    # Source distribution
    # -------------------------------------------------------------

    print("\n" + "-" * 80)
    print("SOURCE DISTRIBUTION")
    print("-" * 80)

    source_counts = (
        df["candidate_source"]
        .value_counts()
    )

    for source, count in source_counts.items():

        print(
            f"{source}: {count:,}"
            f" ({count / len(df) * 100:.2f}%)"
        )

    # -------------------------------------------------------------
    # Country agreement
    # -------------------------------------------------------------

    print("\n" + "-" * 80)
    print("COUNTRY AGREEMENT")
    print("-" * 80)

    country_match = percentage(
        df["country_exact"]
    )

    print(
        f"Exact country agreement: "
        f"{country_match:.2f}%"
    )

    # -------------------------------------------------------------
    # Exact normalized similarities
    # -------------------------------------------------------------

    print("\n" + "-" * 80)
    print("EXACT NORMALIZED AGREEMENT")
    print("-" * 80)

    print(
        f"Name exact: "
        f"{percentage(df['name_exact_normalized']):.2f}%"
    )

    print(
        f"Address exact: "
        f"{percentage(df['address_exact_normalized']):.2f}%"
    )

    # -------------------------------------------------------------
    # Name similarity thresholds
    # -------------------------------------------------------------

    print("\n" + "-" * 80)
    print("NAME SIMILARITY DISTRIBUTION")
    print("-" * 80)

    for threshold in [
        0.50,
        0.60,
        0.70,
        0.75,
        0.80,
        0.85,
        0.90,
        0.95
    ]:

        char_rate = (
            df["name_char_similarity"]
            >= threshold
        ).mean() * 100

        jaccard_rate = (
            df["name_jaccard"]
            >= threshold
        ).mean() * 100

        ngram_rate = (
            df["name_char3_jaccard"]
            >= threshold
        ).mean() * 100

        print(
            f">= {threshold:.2f} : "
            f"char={char_rate:6.2f}% | "
            f"token={jaccard_rate:6.2f}% | "
            f"char3={ngram_rate:6.2f}%"
        )

    # -------------------------------------------------------------
    # Address similarity thresholds
    # -------------------------------------------------------------

    print("\n" + "-" * 80)
    print("ADDRESS SIMILARITY DISTRIBUTION")
    print("-" * 80)

    for threshold in [
        0.30,
        0.40,
        0.50,
        0.60,
        0.70,
        0.75,
        0.80,
        0.90
    ]:

        char_rate = (
            df["address_char_similarity"]
            >= threshold
        ).mean() * 100

        jaccard_rate = (
            df["address_jaccard"]
            >= threshold
        ).mean() * 100

        ngram_rate = (
            df["address_char3_jaccard"]
            >= threshold
        ).mean() * 100

        print(
            f">= {threshold:.2f} : "
            f"char={char_rate:6.2f}% | "
            f"token={jaccard_rate:6.2f}% | "
            f"char3={ngram_rate:6.2f}%"
        )

    # -------------------------------------------------------------
    # Combined evidence
    # -------------------------------------------------------------

    print("\n" + "-" * 80)
    print("COMBINED EVIDENCE")
    print("-" * 80)

    conditions = {

        "name_char >= 0.80":
            df["name_char_similarity"] >= 0.80,

        "name_char >= 0.90":
            df["name_char_similarity"] >= 0.90,

        "name_jaccard >= 0.50":
            df["name_jaccard"] >= 0.50,

        "address_char >= 0.80":
            df["address_char_similarity"] >= 0.80,

        "address_jaccard >= 0.50":
            df["address_jaccard"] >= 0.50,

        "name >= 0.80 AND address >= 0.50":
            (
                (df["name_char_similarity"] >= 0.80)
                &
                (df["address_jaccard"] >= 0.50)
            ),

        "name >= 0.80 OR address >= 0.50":
            (
                (df["name_char_similarity"] >= 0.80)
                |
                (df["address_jaccard"] >= 0.50)
            ),

        "name >= 0.70 OR address >= 0.40":
            (
                (df["name_char_similarity"] >= 0.70)
                |
                (df["address_jaccard"] >= 0.40)
            ),
    }

    for label, mask in conditions.items():

        print(
            f"{label:40s}"
            f"{mask.mean() * 100:7.2f}%"
        )

    # -------------------------------------------------------------
    # Descriptive statistics
    # -------------------------------------------------------------

    print("\n" + "-" * 80)
    print("MEDIAN / MEAN SIMILARITY")
    print("-" * 80)

    metrics = [
        "name_jaccard",
        "name_char_similarity",
        "name_char3_jaccard",
        "address_jaccard",
        "address_char_similarity",
        "address_char3_jaccard",
        "address_common_numbers"
    ]

    for col in metrics:

        print(
            f"{col:30s}"
            f"mean={df[col].mean():.4f}  "
            f"median={df[col].median():.4f}"
        )

    # -------------------------------------------------------------
    # High-evidence missed pairs
    # -------------------------------------------------------------

    print("\n" + "-" * 80)
    print("HIGH-SIMILARITY MISSED PAIRS")
    print("-" * 80)

    high = df[
        (
            df["name_char_similarity"] >= 0.85
        )
        |
        (
            df["address_char_similarity"] >= 0.85
        )
    ].copy()

    print(
        "Missed pairs with strong character similarity:",
        f"{len(high):,}",
        f"({len(high) / len(df) * 100:.2f}%)"
    )

    if len(high) > 0:

        display_columns = [
            "candidate_source",
            "s1_name",
            "candidate_name",
            "s1_address",
            "candidate_address",
            "name_char_similarity",
            "name_jaccard",
            "address_char_similarity",
            "address_jaccard",
            "address_common_numbers",
            "s1_country",
            "candidate_country"
        ]

        print(
            "\nFirst 20 examples:\n"
        )

        print(
            high[
                display_columns
            ]
            .head(20)
            .to_string(index=False)
        )

    print("\n" + "=" * 80)
    print("DIAGNOSTIC ANALYSIS COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
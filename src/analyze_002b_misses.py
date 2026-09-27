from pathlib import Path
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent
INPUT_PATH = (
    BASE_DIR
    / "experiments"
    / "001_baseline"
    / "missed_002b_diagnostics.tsv"
)


def pct(series):
    return round(series.mean() * 100, 2)


def threshold_table(df, column, thresholds):

    rows = []

    for threshold in thresholds:

        rows.append({
            "threshold": threshold,
            "count": int(
                (df[column] >= threshold).sum()
            ),
            "percent": round(
                (df[column] >= threshold).mean() * 100,
                2
            )
        })

    return pd.DataFrame(rows)


def main():

    print("=" * 80)
    print("002B MISSED-PAIR ANALYSIS")
    print("=" * 80)

    df = pd.read_csv(
        INPUT_PATH,
        sep="\t",
        dtype={
            "source1_entity_id": str,
            "candidate_entity_id": str,
            "candidate_source": str
        }
    )

    print(f"\nRows: {len(df):,}")

    # ---------------------------------------------------------
    # Source distribution
    # ---------------------------------------------------------

    print("\n" + "=" * 80)
    print("1. SOURCE DISTRIBUTION")
    print("=" * 80)

    print(
        df["candidate_source"]
        .value_counts()
        .to_string()
    )

    # ---------------------------------------------------------
    # Exact matching
    # ---------------------------------------------------------

    print("\n" + "=" * 80)
    print("2. EXACT MATCH RATES")
    print("=" * 80)

    exact_cols = [
        "name_exact_normalized",
        "address_exact_normalized",
        "country_exact"
    ]

    for col in exact_cols:

        print(
            f"{col}: "
            f"{pct(df[col]):.2f}%"
        )

    # ---------------------------------------------------------
    # Similarity thresholds
    # ---------------------------------------------------------

    print("\n" + "=" * 80)
    print("3. NAME CHARACTER SIMILARITY")
    print("=" * 80)

    print(
        threshold_table(
            df,
            "name_char_similarity",
            [
                0.50,
                0.60,
                0.70,
                0.75,
                0.80,
                0.85,
                0.90,
                0.95
            ]
        ).to_string(index=False)
    )

    print("\n" + "=" * 80)
    print("4. NAME TOKEN / JACCARD")
    print("=" * 80)

    print(
        threshold_table(
            df,
            "name_jaccard",
            [
                0.30,
                0.40,
                0.50,
                0.60,
                0.70,
                0.75,
                0.80,
                0.90
            ]
        ).to_string(index=False)
    )

    print("\n" + "=" * 80)
    print("5. NAME CHAR-3 JACCARD")
    print("=" * 80)

    print(
        threshold_table(
            df,
            "name_char3_jaccard",
            [
                0.30,
                0.40,
                0.50,
                0.60,
                0.70,
                0.80
            ]
        ).to_string(index=False)
    )

    # ---------------------------------------------------------
    # Address
    # ---------------------------------------------------------

    print("\n" + "=" * 80)
    print("6. ADDRESS CHARACTER SIMILARITY")
    print("=" * 80)

    print(
        threshold_table(
            df,
            "address_char_similarity",
            [
                0.30,
                0.40,
                0.50,
                0.60,
                0.70,
                0.75,
                0.80,
                0.85,
                0.90,
                0.95
            ]
        ).to_string(index=False)
    )

    print("\n" + "=" * 80)
    print("7. ADDRESS TOKEN / JACCARD")
    print("=" * 80)

    print(
        threshold_table(
            df,
            "address_jaccard",
            [
                0.30,
                0.40,
                0.50,
                0.60,
                0.70,
                0.75,
                0.80,
                0.90
            ]
        ).to_string(index=False)
    )

    print("\n" + "=" * 80)
    print("8. ADDRESS CHAR-3 JACCARD")
    print("=" * 80)

    print(
        threshold_table(
            df,
            "address_char3_jaccard",
            [
                0.30,
                0.40,
                0.50,
                0.60,
                0.70,
                0.80
            ]
        ).to_string(index=False)
    )

    # ---------------------------------------------------------
    # Numeric evidence
    # ---------------------------------------------------------

    print("\n" + "=" * 80)
    print("9. ADDRESS NUMBER OVERLAP")
    print("=" * 80)

    print(
        df["address_common_numbers"]
        .describe()
        .to_string()
    )

    for n in [0, 1, 2, 3]:

        print(
            f"common_numbers >= {n}: "
            f"{(df['address_common_numbers'] >= n).mean() * 100:.2f}%"
        )

    # ---------------------------------------------------------
    # Combined evidence
    # ---------------------------------------------------------

    print("\n" + "=" * 80)
    print("10. COMBINED EVIDENCE")
    print("=" * 80)

    conditions = {

        "name>=0.80 OR address>=0.50":
            (
                (df["name_char_similarity"] >= 0.80)
                |
                (df["address_char_similarity"] >= 0.50)
            ),

        "name>=0.80 AND address>=0.50":
            (
                (df["name_char_similarity"] >= 0.80)
                &
                (df["address_char_similarity"] >= 0.50)
            ),

        "name>=0.70 OR address>=0.40":
            (
                (df["name_char_similarity"] >= 0.70)
                |
                (df["address_char_similarity"] >= 0.40)
            ),

        "name>=0.85 OR address>=0.85":
            (
                (df["name_char_similarity"] >= 0.85)
                |
                (df["address_char_similarity"] >= 0.85)
            ),

        "name_jaccard>=0.50 OR address_jaccard>=0.50":
            (
                (df["name_jaccard"] >= 0.50)
                |
                (df["address_jaccard"] >= 0.50)
            ),

        "name_char3>=0.50 OR address_char3>=0.50":
            (
                (df["name_char3_jaccard"] >= 0.50)
                |
                (df["address_char3_jaccard"] >= 0.50)
            )
    }

    for label, condition in conditions.items():

        print(
            f"{label}: "
            f"{condition.mean() * 100:.2f}%"
        )

    # ---------------------------------------------------------
    # Strong-name / weak-address
    # ---------------------------------------------------------

    print("\n" + "=" * 80)
    print("11. EVIDENCE COMBINATIONS")
    print("=" * 80)

    combinations = {

        "strong name (>=0.85), weak address (<0.50)":
            (
                (df["name_char_similarity"] >= 0.85)
                &
                (df["address_char_similarity"] < 0.50)
            ),

        "weak name (<0.70), strong address (>=0.85)":
            (
                (df["name_char_similarity"] < 0.70)
                &
                (df["address_char_similarity"] >= 0.85)
            ),

        "strong name (>=0.85), strong address (>=0.85)":
            (
                (df["name_char_similarity"] >= 0.85)
                &
                (df["address_char_similarity"] >= 0.85)
            ),

        "weak name (<0.70), weak address (<0.70)":
            (
                (df["name_char_similarity"] < 0.70)
                &
                (df["address_char_similarity"] < 0.70)
            )
    }

    for label, condition in combinations.items():

        print(
            f"{label}: "
            f"{condition.mean() * 100:.2f}%"
        )

    # ---------------------------------------------------------
    # Source-specific analysis
    # ---------------------------------------------------------

    print("\n" + "=" * 80)
    print("12. S2 vs S3")
    print("=" * 80)

    for source in ["S2", "S3"]:

        subset = df[
            df["candidate_source"] == source
        ]

        print(f"\n--- {source} ---")
        print(f"Rows: {len(subset):,}")

        print(
            f"Name char >= 0.80: "
            f"{(subset['name_char_similarity'] >= 0.80).mean() * 100:.2f}%"
        )

        print(
            f"Name char >= 0.90: "
            f"{(subset['name_char_similarity'] >= 0.90).mean() * 100:.2f}%"
        )

        print(
            f"Address char >= 0.80: "
            f"{(subset['address_char_similarity'] >= 0.80).mean() * 100:.2f}%"
        )

        print(
            f"Address char >= 0.90: "
            f"{(subset['address_char_similarity'] >= 0.90).mean() * 100:.2f}%"
        )

        print(
            f"Name >=0.80 OR address >=0.50: "
            f"{conditions['name>=0.80 OR address>=0.50'][subset.index].mean() * 100:.2f}%"
        )

    # ---------------------------------------------------------
    # Descriptive statistics
    # ---------------------------------------------------------

    print("\n" + "=" * 80)
    print("13. DESCRIPTIVE STATISTICS")
    print("=" * 80)

    numeric_cols = [
        "name_jaccard",
        "name_char_similarity",
        "name_char3_jaccard",
        "address_jaccard",
        "address_char_similarity",
        "address_char3_jaccard",
        "address_common_numbers"
    ]

    print(
        df[numeric_cols]
        .describe()
        .T
        .round(4)
        .to_string()
    )

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
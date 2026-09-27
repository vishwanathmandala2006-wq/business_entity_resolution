from pathlib import Path
import re
from difflib import SequenceMatcher

import pandas as pd


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

TRAIN_DIR = BASE_DIR / "dataset" / "train"
EXP_DIR = BASE_DIR / "experiments" / "001_baseline"

S1_PATH = TRAIN_DIR / "train_source1.tsv"
S2_PATH = TRAIN_DIR / "train_source2.tsv"
S3_PATH = TRAIN_DIR / "train_source3.tsv"

MISSED_PATH = EXP_DIR / "missed_002b_pairs.tsv"

OUTPUT_PATH = EXP_DIR / "missed_002b_diagnostics.tsv"

SAMPLE_SIZE = 10000


# ---------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------

def normalize_text(value):

    if pd.isna(value):
        return ""

    value = str(value).lower().strip()

    value = re.sub(
        r"[^a-z0-9\s]",
        " ",
        value
    )

    value = re.sub(
        r"\s+",
        " ",
        value
    )

    return value.strip()


def tokenize(value):

    value = normalize_text(value)

    if not value:
        return set()

    return set(value.split())


def jaccard(a, b):

    if not a or not b:
        return 0.0

    union = a | b

    if not union:
        return 0.0

    return len(a & b) / len(union)


def character_ngram_set(value, n=3):

    value = normalize_text(value)

    if len(value) < n:
        return {value} if value else set()

    return {
        value[i:i+n]
        for i in range(len(value) - n + 1)
    }


def extract_numbers(value):

    if pd.isna(value):
        return set()

    return set(
        re.findall(
            r"\d+",
            str(value)
        )
    )


def sequence_ratio(a, b):

    if not a or not b:
        return 0.0

    return SequenceMatcher(
        None,
        a,
        b
    ).ratio()


# ---------------------------------------------------------------------
# Load only required records
# ---------------------------------------------------------------------

def load_required_records(
    path,
    required_ids,
    source_name
):

    print(
        f"\nLoading required {source_name} records..."
    )

    records = {}

    usecols = [
        "entity_id",
        "business_name",
        "business_address",
        "country"
    ]

    for chunk in pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        usecols=usecols,
        chunksize=250_000
    ):

        matched = chunk[
            chunk["entity_id"].isin(required_ids)
        ]

        for row in matched.itertuples(index=False):

            records[row.entity_id] = {
                "business_name": row.business_name,
                "business_address": row.business_address,
                "country": row.country
            }

        if len(records) == len(required_ids):
            break

    print(
        f"{source_name} records retrieved:",
        f"{len(records):,}",
        "/",
        f"{len(required_ids):,}"
    )

    return records


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():

    print("=" * 80)
    print("002B MISSED TRUE-PAIR DIAGNOSTICS")
    print("=" * 80)

    # -------------------------------------------------------------
    # Load already-extracted missed pairs
    # -------------------------------------------------------------

    print("\nLoading missed 002B pairs...")

    missed = pd.read_csv(
        MISSED_PATH,
        sep="\t",
        dtype=str,
        keep_default_na=False
    )

    print(
        "Total missed pairs:",
        f"{len(missed):,}"
    )

    # -------------------------------------------------------------
    # Sample
    # -------------------------------------------------------------

    if len(missed) > SAMPLE_SIZE:

        missed_sample = missed.sample(
            n=SAMPLE_SIZE,
            random_state=42
        ).reset_index(drop=True)

    else:

        missed_sample = missed.copy()

    # Determine source from candidate ID
    missed_sample["candidate_source"] = (
        missed_sample["candidate_entity_id"]
        .str.startswith("S2-")
        .map({
            True: "S2",
            False: "S3"
        })
    )

    print(
        "Diagnostic sample:",
        f"{len(missed_sample):,}"
    )

    print("\nSource distribution:")

    print(
        missed_sample["candidate_source"]
        .value_counts()
        .to_string()
    )

    # -------------------------------------------------------------
    # Required IDs
    # -------------------------------------------------------------

    s1_ids = set(
        missed_sample["source1_entity_id"]
    )

    s2_ids = set(
        missed_sample.loc[
            missed_sample["candidate_source"] == "S2",
            "candidate_entity_id"
        ]
    )

    s3_ids = set(
        missed_sample.loc[
            missed_sample["candidate_source"] == "S3",
            "candidate_entity_id"
        ]
    )

    print(
        "\nRequired S1 records:",
        f"{len(s1_ids):,}"
    )

    print(
        "Required S2 records:",
        f"{len(s2_ids):,}"
    )

    print(
        "Required S3 records:",
        f"{len(s3_ids):,}"
    )

    # -------------------------------------------------------------
    # Load records
    # -------------------------------------------------------------

    s1_records = load_required_records(
        S1_PATH,
        s1_ids,
        "S1"
    )

    s2_records = load_required_records(
        S2_PATH,
        s2_ids,
        "S2"
    )

    s3_records = load_required_records(
        S3_PATH,
        s3_ids,
        "S3"
    )

    # -------------------------------------------------------------
    # Diagnostics
    # -------------------------------------------------------------

    diagnostic_rows = []

    for row in missed_sample.itertuples(index=False):

        s1 = s1_records.get(
            row.source1_entity_id
        )

        if row.candidate_source == "S2":

            candidate = s2_records.get(
                row.candidate_entity_id
            )

        else:

            candidate = s3_records.get(
                row.candidate_entity_id
            )

        if s1 is None or candidate is None:
            continue

        s1_name = normalize_text(
            s1["business_name"]
        )

        candidate_name = normalize_text(
            candidate["business_name"]
        )

        s1_address = normalize_text(
            s1["business_address"]
        )

        candidate_address = normalize_text(
            candidate["business_address"]
        )

        name_tokens_s1 = tokenize(
            s1["business_name"]
        )

        name_tokens_candidate = tokenize(
            candidate["business_name"]
        )

        address_tokens_s1 = tokenize(
            s1["business_address"]
        )

        address_tokens_candidate = tokenize(
            candidate["business_address"]
        )

        name_ngrams_s1 = character_ngram_set(
            s1["business_name"]
        )

        name_ngrams_candidate = character_ngram_set(
            candidate["business_name"]
        )

        address_ngrams_s1 = character_ngram_set(
            s1["business_address"]
        )

        address_ngrams_candidate = character_ngram_set(
            candidate["business_address"]
        )

        s1_numbers = extract_numbers(
            s1["business_address"]
        )

        candidate_numbers = extract_numbers(
            candidate["business_address"]
        )

        common_numbers = (
            s1_numbers & candidate_numbers
        )

        diagnostic_rows.append(
            {
                "source1_entity_id":
                    row.source1_entity_id,

                "candidate_entity_id":
                    row.candidate_entity_id,

                "candidate_source":
                    row.candidate_source,

                "s1_name":
                    s1["business_name"],

                "candidate_name":
                    candidate["business_name"],

                "s1_address":
                    s1["business_address"],

                "candidate_address":
                    candidate["business_address"],

                "s1_country":
                    s1["country"],

                "candidate_country":
                    candidate["country"],

                "name_exact_normalized":
                    int(
                        s1_name == candidate_name
                        and bool(s1_name)
                    ),

                "name_jaccard":
                    round(
                        jaccard(
                            name_tokens_s1,
                            name_tokens_candidate
                        ),
                        4
                    ),

                "name_char_similarity":
                    round(
                        sequence_ratio(
                            s1_name,
                            candidate_name
                        ),
                        4
                    ),

                "name_char3_jaccard":
                    round(
                        jaccard(
                            name_ngrams_s1,
                            name_ngrams_candidate
                        ),
                        4
                    ),

                "address_exact_normalized":
                    int(
                        s1_address == candidate_address
                        and bool(s1_address)
                    ),

                "address_jaccard":
                    round(
                        jaccard(
                            address_tokens_s1,
                            address_tokens_candidate
                        ),
                        4
                    ),

                "address_char_similarity":
                    round(
                        sequence_ratio(
                            s1_address,
                            candidate_address
                        ),
                        4
                    ),

                "address_char3_jaccard":
                    round(
                        jaccard(
                            address_ngrams_s1,
                            address_ngrams_candidate
                        ),
                        4
                    ),

                "address_common_numbers":
                    len(common_numbers),

                "country_exact":
                    int(
                        normalize_text(
                            s1["country"]
                        )
                        ==
                        normalize_text(
                            candidate["country"]
                        )
                    )
            }
        )

    # -------------------------------------------------------------
    # Save
    # -------------------------------------------------------------

    diagnostics = pd.DataFrame(
        diagnostic_rows
    )

    diagnostics.to_csv(
        OUTPUT_PATH,
        sep="\t",
        index=False
    )

    print(
        "\nDiagnostic rows saved:",
        f"{len(diagnostics):,}"
    )

    print(
        "\nOutput:",
        OUTPUT_PATH
    )

    print("\n" + "=" * 80)
    print("002B MISSED-PAIR DIAGNOSTICS COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
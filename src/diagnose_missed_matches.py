from pathlib import Path
import re

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent

TRAIN_DIR = BASE_DIR / "dataset" / "train"
EXP_DIR = BASE_DIR / "experiments" / "001_baseline"

S1_PATH = TRAIN_DIR / "train_source1.tsv"
S2_PATH = TRAIN_DIR / "train_source2.tsv"
S3_PATH = TRAIN_DIR / "train_source3.tsv"

GT_PATH = EXP_DIR / "validation_ground_truth.tsv"
CANDIDATE_PATH = EXP_DIR / "structural_candidates.tsv"

OUTPUT_PATH = EXP_DIR / "missed_pair_diagnostics.tsv"

SAMPLE_SIZE = 5000


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

    # Standard-library similarity.
    # No external dependency required.
    from difflib import SequenceMatcher

    if not a or not b:
        return 0.0

    return SequenceMatcher(
        None,
        a,
        b
    ).ratio()


# ---------------------------------------------------------------------
# Load records by IDs
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

    # Read in chunks so we don't unnecessarily keep
    # the complete 5M-row source in memory.
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
    print("MISSED TRUE-PAIR DIAGNOSTICS")
    print("=" * 80)

    # -------------------------------------------------------------
    # Load validation GT
    # -------------------------------------------------------------

    print("\nLoading validation ground truth...")

    gt = pd.read_csv(
        GT_PATH,
        sep="\t",
        dtype=str,
        keep_default_na=False
    )

    # -------------------------------------------------------------
    # Load candidate pairs
    # -------------------------------------------------------------

    print("Loading structural candidates...")

    candidates = pd.read_csv(
        CANDIDATE_PATH,
        sep="\t",
        dtype=str,
        keep_default_na=False
    )

    candidate_keys = set(
        zip(
            candidates["source1_entity_id"],
            candidates["candidate_entity_id"]
        )
    )

    # -------------------------------------------------------------
    # Identify missed true pairs
    # -------------------------------------------------------------

    missed_pairs = []

    for row in gt.itertuples(index=False):

        s1_id = row.source1_entity_id

        if not row.matched_entity_ids.strip():
            continue

        true_ids = [
            x.strip()
            for x in row.matched_entity_ids.split(",")
            if x.strip()
        ]

        for true_id in true_ids:

            if (
                s1_id,
                true_id
            ) not in candidate_keys:

                source = (
                    "S2"
                    if true_id.startswith("S2-")
                    else "S3"
                )

                missed_pairs.append(
                    (
                        s1_id,
                        true_id,
                        source
                    )
                )

    print(
        "\nTotal missed true pairs:",
        f"{len(missed_pairs):,}"
    )

    # -------------------------------------------------------------
    # Sample missed pairs
    # -------------------------------------------------------------

    if len(missed_pairs) > SAMPLE_SIZE:

        missed_sample = pd.DataFrame(
            missed_pairs,
            columns=[
                "s1_id",
                "candidate_id",
                "candidate_source"
            ]
        ).sample(
            n=SAMPLE_SIZE,
            random_state=42
        )

    else:

        missed_sample = pd.DataFrame(
            missed_pairs,
            columns=[
                "s1_id",
                "candidate_id",
                "candidate_source"
            ]
        )

    # -------------------------------------------------------------
    # Required IDs
    # -------------------------------------------------------------

    s1_ids = set(
        missed_sample["s1_id"]
    )

    s2_ids = set(
        missed_sample.loc[
            missed_sample["candidate_source"] == "S2",
            "candidate_id"
        ]
    )

    s3_ids = set(
        missed_sample.loc[
            missed_sample["candidate_source"] == "S3",
            "candidate_id"
        ]
    )

    print(
        "\nSampled missed pairs:",
        f"{len(missed_sample):,}"
    )

    print(
        "Required S1 records:",
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
    # Load only required records
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
    # Build diagnostics
    # -------------------------------------------------------------

    diagnostic_rows = []

    for row in missed_sample.itertuples(index=False):

        s1 = s1_records.get(row.s1_id)

        if row.candidate_source == "S2":
            candidate = s2_records.get(
                row.candidate_id
            )
        else:
            candidate = s3_records.get(
                row.candidate_id
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
                "source1_entity_id": row.s1_id,
                "candidate_entity_id": row.candidate_id,
                "candidate_source": row.candidate_source,

                "s1_name": s1["business_name"],
                "candidate_name": candidate["business_name"],

                "s1_address": s1["business_address"],
                "candidate_address": candidate["business_address"],

                "s1_country": s1["country"],
                "candidate_country": candidate["country"],

                "name_exact_normalized": int(
                    s1_name == candidate_name
                    and bool(s1_name)
                ),

                "name_jaccard": round(
                    jaccard(
                        name_tokens_s1,
                        name_tokens_candidate
                    ),
                    4
                ),

                "name_char_similarity": round(
                    sequence_ratio(
                        s1_name,
                        candidate_name
                    ),
                    4
                ),

                "name_char3_jaccard": round(
                    jaccard(
                        name_ngrams_s1,
                        name_ngrams_candidate
                    ),
                    4
                ),

                "address_exact_normalized": int(
                    s1_address == candidate_address
                    and bool(s1_address)
                ),

                "address_jaccard": round(
                    jaccard(
                        address_tokens_s1,
                        address_tokens_candidate
                    ),
                    4
                ),

                "address_char_similarity": round(
                    sequence_ratio(
                        s1_address,
                        candidate_address
                    ),
                    4
                ),

                "address_char3_jaccard": round(
                    jaccard(
                        address_ngrams_s1,
                        address_ngrams_candidate
                    ),
                    4
                ),

                "address_common_numbers": len(
                    common_numbers
                ),

                "country_exact": int(
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
    print("MISSED-PAIR DIAGNOSTICS COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
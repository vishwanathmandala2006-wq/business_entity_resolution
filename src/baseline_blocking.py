from pathlib import Path
from collections import defaultdict
import re
import time

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent

TRAIN_DIR = BASE_DIR / "dataset" / "train"
EXP_DIR = BASE_DIR / "experiments" / "001_baseline"

S1_PATH = TRAIN_DIR / "train_source1.tsv"
S2_PATH = TRAIN_DIR / "train_source2.tsv"
S3_PATH = TRAIN_DIR / "train_source3.tsv"

VALIDATION_GT_PATH = (
    EXP_DIR / "validation_ground_truth.tsv"
)

OUTPUT_PATH = (
    EXP_DIR / "baseline_candidates.tsv"
)


# ---------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------

def normalize_text(value):
    """
    Conservative normalization.

    We deliberately do NOT perform aggressive transliteration,
    synonym replacement, abbreviation expansion, etc. yet.
    Those will be separate experiments.
    """

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


def make_key(value):
    value = normalize_text(value)

    if not value:
        return ""

    return value


# ---------------------------------------------------------------------
# Build inverted index
# ---------------------------------------------------------------------

def build_indexes(path, source_name):

    print(f"\nLoading {source_name}...")

    df = pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        usecols=[
            "entity_id",
            "business_name",
            "business_address",
            "country"
        ]
    )

    print(
        f"{source_name} rows:",
        f"{len(df):,}"
    )

    name_index = defaultdict(list)
    address_index = defaultdict(list)
    name_country_index = defaultdict(list)
    address_country_index = defaultdict(list)

    for row in df.itertuples(index=False):

        entity_id = row.entity_id

        name = make_key(row.business_name)
        address = make_key(row.business_address)
        country = make_key(row.country)

        if name:
            name_index[name].append(entity_id)

            if country:
                name_country_index[
                    (name, country)
                ].append(entity_id)

        if address:
            address_index[address].append(entity_id)

            if country:
                address_country_index[
                    (address, country)
                ].append(entity_id)

    print(
        "Unique normalized names:",
        f"{len(name_index):,}"
    )

    print(
        "Unique normalized addresses:",
        f"{len(address_index):,}"
    )

    return {
        "name": name_index,
        "address": address_index,
        "name_country": name_country_index,
        "address_country": address_country_index,
    }


# ---------------------------------------------------------------------
# Candidate generation
# ---------------------------------------------------------------------

def generate_candidates(
    s1_df,
    indexes,
    source_name
):

    rows = []

    for row in s1_df.itertuples(index=False):

        s1_id = row.entity_id

        name = make_key(row.business_name)
        address = make_key(row.business_address)
        country = make_key(row.country)

        candidates = {}

        # -------------------------------------------------------------
        # Rule 1: exact normalized name
        # -------------------------------------------------------------

        if name:

            for entity_id in indexes["name"].get(
                name,
                []
            ):

                candidates.setdefault(
                    entity_id,
                    set()
                ).add("NAME")

        # -------------------------------------------------------------
        # Rule 2: exact normalized address
        # -------------------------------------------------------------

        if address:

            for entity_id in indexes["address"].get(
                address,
                []
            ):

                candidates.setdefault(
                    entity_id,
                    set()
                ).add("ADDRESS")

        # -------------------------------------------------------------
        # Rule 3: exact normalized name + country
        # -------------------------------------------------------------

        if name and country:

            for entity_id in indexes[
                "name_country"
            ].get(
                (name, country),
                []
            ):

                candidates.setdefault(
                    entity_id,
                    set()
                ).add("NAME_COUNTRY")

        # -------------------------------------------------------------
        # Rule 4: exact normalized address + country
        # -------------------------------------------------------------

        if address and country:

            for entity_id in indexes[
                "address_country"
            ].get(
                (address, country),
                []
            ):

                candidates.setdefault(
                    entity_id,
                    set()
                ).add("ADDRESS_COUNTRY")

        # -------------------------------------------------------------
        # Save candidate information
        # -------------------------------------------------------------

        for candidate_id, rules in candidates.items():

            rows.append(
                {
                    "source1_entity_id": s1_id,
                    "candidate_entity_id": candidate_id,
                    "candidate_source": source_name,
                    "blocking_rules": "|".join(
                        sorted(rules)
                    )
                }
            )

    return rows


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():

    start = time.time()

    print("=" * 80)
    print("BASELINE BLOCKING")
    print("=" * 80)

    EXP_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # -------------------------------------------------------------
    # Load validation S1 IDs
    # -------------------------------------------------------------

    validation_gt = pd.read_csv(
        VALIDATION_GT_PATH,
        sep="\t",
        dtype=str,
        keep_default_na=False
    )

    validation_ids = set(
        validation_gt[
            "source1_entity_id"
        ]
    )

    print(
        "\nValidation S1 IDs:",
        f"{len(validation_ids):,}"
    )

    # -------------------------------------------------------------
    # Load S1
    # -------------------------------------------------------------

    print("\nLoading S1...")

    s1 = pd.read_csv(
        S1_PATH,
        sep="\t",
        dtype=str,
        keep_default_na=False
    )

    s1 = s1[
        s1["entity_id"].isin(
            validation_ids
        )
    ].copy()

    print(
        "Validation S1 records loaded:",
        f"{len(s1):,}"
    )

    # -------------------------------------------------------------
    # Build S2 indexes
    # -------------------------------------------------------------

    s2_indexes = build_indexes(
        S2_PATH,
        "S2"
    )

    # -------------------------------------------------------------
    # Build S3 indexes
    # -------------------------------------------------------------

    s3_indexes = build_indexes(
        S3_PATH,
        "S3"
    )

    # -------------------------------------------------------------
    # Generate candidates
    # -------------------------------------------------------------

    print("\nGenerating S2 candidates...")

    s2_candidates = generate_candidates(
        s1,
        s2_indexes,
        "S2"
    )

    print(
        "S2 candidate pairs:",
        f"{len(s2_candidates):,}"
    )

    print("\nGenerating S3 candidates...")

    s3_candidates = generate_candidates(
        s1,
        s3_indexes,
        "S3"
    )

    print(
        "S3 candidate pairs:",
        f"{len(s3_candidates):,}"
    )

    # -------------------------------------------------------------
    # Combine
    # -------------------------------------------------------------

    all_candidates = (
        s2_candidates +
        s3_candidates
    )

    candidates_df = pd.DataFrame(
        all_candidates
    )

    if len(candidates_df) > 0:

        candidates_df = (
            candidates_df
            .drop_duplicates(
                subset=[
                    "source1_entity_id",
                    "candidate_entity_id"
                ]
            )
            .sort_values(
                [
                    "source1_entity_id",
                    "candidate_source",
                    "candidate_entity_id"
                ]
            )
        )

    candidates_df.to_csv(
        OUTPUT_PATH,
        sep="\t",
        index=False
    )

    print(
        "\nFinal unique candidate pairs:",
        f"{len(candidates_df):,}"
    )

    print(
        "\nOutput:",
        OUTPUT_PATH
    )

    print(
        "\nRuntime:",
        f"{time.time() - start:.2f} seconds"
    )

    print("\n" + "=" * 80)
    print("BASELINE BLOCKING COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
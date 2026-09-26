from pathlib import Path
from collections import defaultdict, Counter
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
    EXP_DIR / "structural_candidates.tsv"
)


# ---------------------------------------------------------------------
# Blocking limits
# ---------------------------------------------------------------------

# A key occurring more frequently than this is considered
# too common to use as a blocking key.
MAX_KEY_FREQUENCY = 50

# Prevent one S1 record from exploding because of a single
# blocking route.
MAX_CANDIDATES_PER_KEY = 100


# ---------------------------------------------------------------------
# Basic normalization
# ---------------------------------------------------------------------

def basic_normalize(value):

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


# ---------------------------------------------------------------------
# Name normalization
# ---------------------------------------------------------------------

LEGAL_REPLACEMENTS = [
    ("private limited", "pvt"),
    ("private ltd", "pvt"),
    ("pvt limited", "pvt"),
    ("pvt ltd", "pvt"),
    ("incorporated", "inc"),
    ("corporation", "corp"),
    ("company", "co"),
    ("limited", "ltd"),
]


def structural_name(value):

    text = basic_normalize(value)

    if not text:
        return ""

    for old, new in LEGAL_REPLACEMENTS:

        text = re.sub(
            r"\b" + re.escape(old) + r"\b",
            new,
            text
        )

    return re.sub(
        r"\s+",
        " ",
        text
    ).strip()


def name_without_suffix(value):

    text = structural_name(value)

    if not text:
        return ""

    suffixes = {
        "pvt",
        "ltd",
        "llc",
        "llp",
        "inc",
        "corp",
        "co"
    }

    tokens = text.split()

    while tokens and tokens[-1] in suffixes:
        tokens.pop()

    return " ".join(tokens)


def name_compact(value):

    return re.sub(
        r"[^a-z0-9]",
        "",
        name_without_suffix(value)
    )


def name_sorted(value):

    text = name_without_suffix(value)

    if not text:
        return ""

    return " ".join(
        sorted(text.split())
    )


# ---------------------------------------------------------------------
# Address normalization
# ---------------------------------------------------------------------

ADDRESS_REPLACEMENTS = [
    ("street", "st"),
    ("road", "rd"),
    ("avenue", "ave"),
    ("boulevard", "blvd"),
    ("highway", "hwy"),
    ("lane", "ln"),
    ("drive", "dr"),
    ("circle", "cir"),
    ("parkway", "pkwy"),
    ("place", "pl"),
    ("suite", "ste"),
    ("floor", "fl"),
]


def structural_address(value):

    text = basic_normalize(value)

    if not text:
        return ""

    for old, new in ADDRESS_REPLACEMENTS:

        text = re.sub(
            r"\b" + re.escape(old) + r"\b",
            new,
            text
        )

    return re.sub(
        r"\s+",
        " ",
        text
    ).strip()


def address_compact(value):

    return re.sub(
        r"[^a-z0-9]",
        "",
        structural_address(value)
    )


def address_number_key(value):

    text = structural_address(value)

    numbers = re.findall(
        r"\d+",
        text
    )

    # Only useful when there are at least two
    # numerical components.
    if len(numbers) < 2:
        return ""

    return "|".join(
        sorted(set(numbers))
    )


# ---------------------------------------------------------------------
# Generate keys
# ---------------------------------------------------------------------

def get_keys(name, address):

    return {
        "NAME_STRUCT": structural_name(name),

        "NAME_NOSUFFIX": name_without_suffix(name),

        "NAME_COMPACT": name_compact(name),

        "NAME_SORTED": name_sorted(name),

        "ADDRESS_STRUCT": structural_address(address),

        "ADDRESS_COMPACT": address_compact(address),

        "ADDRESS_NUMBERS": address_number_key(address),
    }


# ---------------------------------------------------------------------
# First pass:
# Count key frequencies
# ---------------------------------------------------------------------

def count_keys(path, source_name):

    print(
        f"\nCounting blocking keys for {source_name}..."
    )

    counters = {
        key_type: Counter()
        for key_type in [
            "NAME_STRUCT",
            "NAME_NOSUFFIX",
            "NAME_COMPACT",
            "NAME_SORTED",
            "ADDRESS_STRUCT",
            "ADDRESS_COMPACT",
            "ADDRESS_NUMBERS",
        ]
    }

    total = 0

    for chunk in pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        usecols=[
            "entity_id",
            "business_name",
            "business_address",
        ],
        chunksize=250_000
    ):

        total += len(chunk)

        for row in chunk.itertuples(index=False):

            keys = get_keys(
                row.business_name,
                row.business_address
            )

            for key_type, key in keys.items():

                if key:
                    counters[
                        key_type
                    ][key] += 1

    print(
        f"{source_name} rows processed:",
        f"{total:,}"
    )

    return counters


# ---------------------------------------------------------------------
# Second pass:
# Build only rare-key indexes
# ---------------------------------------------------------------------

def build_rare_indexes(
    path,
    source_name,
    counters
):

    print(
        f"\nBuilding rare-key indexes for {source_name}..."
    )

    indexes = {
        key_type: defaultdict(list)
        for key_type in counters
    }

    total_indexed = 0

    for chunk in pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        usecols=[
            "entity_id",
            "business_name",
            "business_address",
        ],
        chunksize=250_000
    ):

        for row in chunk.itertuples(index=False):

            keys = get_keys(
                row.business_name,
                row.business_address
            )

            for key_type, key in keys.items():

                if not key:
                    continue

                frequency = counters[
                    key_type
                ][key]

                if (
                    frequency <= MAX_KEY_FREQUENCY
                    and frequency <= MAX_CANDIDATES_PER_KEY
                ):

                    indexes[
                        key_type
                    ][key].append(
                        row.entity_id
                    )

                    total_indexed += 1

    print(
        f"{source_name} indexed key assignments:",
        f"{total_indexed:,}"
    )

    for key_type in indexes:

        print(
            f"  {key_type:20s}: "
            f"{len(indexes[key_type]):,} rare keys"
        )

    return indexes


# ---------------------------------------------------------------------
# Generate candidates
# ---------------------------------------------------------------------

def generate_candidates(
    s1,
    indexes,
    source_name
):

    output = []

    for row in s1.itertuples(index=False):

        keys = get_keys(
            row.business_name,
            row.business_address
        )

        candidates = {}

        for key_type, key in keys.items():

            if not key:
                continue

            entity_ids = indexes[
                key_type
            ].get(key, [])

            for entity_id in entity_ids:

                candidates.setdefault(
                    entity_id,
                    set()
                ).add(key_type)

        for entity_id, rules in candidates.items():

            output.append(
                {
                    "source1_entity_id": row.entity_id,
                    "candidate_entity_id": entity_id,
                    "candidate_source": source_name,
                    "blocking_rules": "|".join(
                        sorted(rules)
                    )
                }
            )

    return output


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():

    start = time.time()

    print("=" * 80)
    print("EXPERIMENT 002A - SELECTIVE STRUCTURAL BLOCKING")
    print("=" * 80)

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
        "Validation S1 records:",
        f"{len(s1):,}"
    )

    # -------------------------------------------------------------
    # S2
    # -------------------------------------------------------------

    s2_counters = count_keys(
        S2_PATH,
        "S2"
    )

    s2_indexes = build_rare_indexes(
        S2_PATH,
        "S2",
        s2_counters
    )

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

    # -------------------------------------------------------------
    # S3
    # -------------------------------------------------------------

    s3_counters = count_keys(
        S3_PATH,
        "S3"
    )

    s3_indexes = build_rare_indexes(
        S3_PATH,
        "S3",
        s3_counters
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

    candidates_df = pd.DataFrame(
        s2_candidates + s3_candidates
    )

    if len(candidates_df):

        candidates_df = (
            candidates_df
            .groupby(
                [
                    "source1_entity_id",
                    "candidate_entity_id",
                    "candidate_source"
                ],
                as_index=False
            )["blocking_rules"]
            .agg("|".join)
        )

        candidates_df["blocking_rules"] = (
            candidates_df[
                "blocking_rules"
            ]
            .apply(
                lambda x: "|".join(
                    sorted(
                        set(
                            x.split("|")
                        )
                    )
                )
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
    print("EXPERIMENT 002A COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
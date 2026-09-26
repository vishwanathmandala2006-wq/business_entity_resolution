from pathlib import Path
import re
import sys
import gc

import pandas as pd
from rapidfuzz.fuzz import ratio
from rapidfuzz.distance import Jaro, JaroWinkler


# =============================================================================
# PATHS
# =============================================================================

BASE_DIR = Path(__file__).resolve().parents[1]

TRAIN_DIR = BASE_DIR / "dataset" / "train"

PAIR_PATH = (
    BASE_DIR
    / "experiments"
    / "005_ml_training"
    / "training_pairs.tsv"
)

OUTPUT_PATH = (
    BASE_DIR
    / "experiments"
    / "005_ml_training"
    / "training_pair_features.tsv"
)

CACHE_DIR = (
    BASE_DIR
    / "experiments"
    / "005_ml_training"
    / "entity_cache"
)


# =============================================================================
# CONFIGURATION
# =============================================================================

PAIR_READ_CHUNK_SIZE = 50_000
ENTITY_SCAN_CHUNK_SIZE = 250_000


# =============================================================================
# TEXT NORMALIZATION
# =============================================================================

def normalize_text(value):
    if pd.isna(value):
        return ""

    value = str(value).lower()

    value = re.sub(
        r"[^a-z0-9]+",
        " ",
        value
    )

    value = re.sub(
        r"\s+",
        " ",
        value
    ).strip()

    return value


def normalize_address(value):
    if pd.isna(value):
        return ""

    value = str(value).lower()

    value = re.sub(
        r"[^a-z0-9]+",
        " ",
        value
    )

    value = re.sub(
        r"\s+",
        " ",
        value
    ).strip()

    return value


# =============================================================================
# NAME NORMALIZATION
# =============================================================================

LEGAL_SUFFIXES = {
    "inc",
    "incorporated",
    "corp",
    "corporation",
    "company",
    "co",
    "limited",
    "ltd",
    "llc",
    "llp",
    "plc",
    "pvt",
    "private",
    "public",
    "industries",
    "industry",
    "enterprises",
    "enterprise",
}


def remove_legal_suffixes(text):

    if not text:
        return ""

    tokens = text.split()

    tokens = [
        token
        for token in tokens
        if token not in LEGAL_SUFFIXES
    ]

    return " ".join(tokens)


def compact_name(text):

    if not text:
        return ""

    return text.replace(" ", "")


def sorted_name(text):

    if not text:
        return ""

    return " ".join(
        sorted(text.split())
    )


# =============================================================================
# TOKEN FEATURES
# =============================================================================

def token_set(text):

    if not text:
        return set()

    return set(text.split())


def token_jaccard(a, b):

    a_tokens = token_set(a)
    b_tokens = token_set(b)

    if not a_tokens and not b_tokens:
        return 1.0

    if not a_tokens or not b_tokens:
        return 0.0

    union = a_tokens | b_tokens

    if not union:
        return 0.0

    return (
        len(a_tokens & b_tokens)
        / len(union)
    )


def token_overlap(a, b):

    a_tokens = token_set(a)
    b_tokens = token_set(b)

    if not a_tokens or not b_tokens:
        return 0.0

    denominator = min(
        len(a_tokens),
        len(b_tokens),
    )

    if denominator == 0:
        return 0.0

    return (
        len(a_tokens & b_tokens)
        / denominator
    )


# =============================================================================
# ADDRESS NUMBER FEATURES
# =============================================================================

def extract_numbers(text):

    if not text:
        return set()

    return set(
        re.findall(
            r"\d+",
            text
        )
    )


def number_features(a, b):

    a_numbers = extract_numbers(a)
    b_numbers = extract_numbers(b)

    if not a_numbers or not b_numbers:
        return 0.0, 0.0, 0.0

    intersection = (
        a_numbers & b_numbers
    )

    overlap = (
        len(intersection)
        / min(
            len(a_numbers),
            len(b_numbers),
        )
    )

    match = (
        1.0
        if intersection
        else 0.0
    )

    conflict = (
        1.0
        if not intersection
        else 0.0
    )

    return (
        overlap,
        match,
        conflict,
    )


# =============================================================================
# POSTAL FEATURES
# =============================================================================

def extract_postal_codes(text):

    if not text:
        return set()

    return set(
        re.findall(
            r"\b\d{4,6}\b",
            text
        )
    )


def postal_features(a, b):

    a_postal = extract_postal_codes(a)
    b_postal = extract_postal_codes(b)

    if not a_postal or not b_postal:
        return 0.0, 0.0

    match = (
        1.0
        if a_postal & b_postal
        else 0.0
    )

    conflict = (
        1.0
        if not (a_postal & b_postal)
        else 0.0
    )

    return (
        match,
        conflict,
    )


# =============================================================================
# NAME FEATURES
# =============================================================================

def name_features(
    name1,
    name2,
):

    n1 = normalize_text(name1)
    n2 = normalize_text(name2)

    n1_no_suffix = (
        remove_legal_suffixes(n1)
    )

    n2_no_suffix = (
        remove_legal_suffixes(n2)
    )

    n1_compact = compact_name(n1)
    n2_compact = compact_name(n2)

    n1_sorted = sorted_name(n1)
    n2_sorted = sorted_name(n2)

    return {

        "name_exact": (
            1.0
            if n1 and n1 == n2
            else 0.0
        ),

        "name_char_ratio": (
            ratio(n1, n2) / 100.0
            if n1 and n2
            else 0.0
        ),

        "name_jaro": (
            Jaro.normalized_similarity(
                n1,
                n2,
            )
            if n1 and n2
            else 0.0
        ),

        "name_jaro_winkler": (
            JaroWinkler.normalized_similarity(
                n1,
                n2,
            )
            if n1 and n2
            else 0.0
        ),

        "name_token_jaccard": (
            token_jaccard(
                n1,
                n2,
            )
        ),

        "name_token_overlap": (
            token_overlap(
                n1,
                n2,
            )
        ),

        "name_length_diff": (
            abs(
                len(n1) - len(n2)
            )
        ),

        "name_token_count_diff": (
            abs(
                len(n1.split())
                - len(n2.split())
            )
        ),

        "name_no_suffix_ratio": (
            ratio(
                n1_no_suffix,
                n2_no_suffix,
            ) / 100.0
            if n1_no_suffix and n2_no_suffix
            else 0.0
        ),

        "name_compact_ratio": (
            ratio(
                n1_compact,
                n2_compact,
            ) / 100.0
            if n1_compact and n2_compact
            else 0.0
        ),

        "name_sorted_ratio": (
            ratio(
                n1_sorted,
                n2_sorted,
            ) / 100.0
            if n1_sorted and n2_sorted
            else 0.0
        ),
    }


# =============================================================================
# ADDRESS FEATURES
# =============================================================================

def address_features(
    address1,
    address2,
):

    a1 = normalize_address(address1)
    a2 = normalize_address(address2)

    (
        number_overlap,
        number_match,
        number_conflict,
    ) = number_features(
        a1,
        a2,
    )

    (
        postal_match,
        postal_conflict,
    ) = postal_features(
        a1,
        a2,
    )

    return {

        "address_exact": (
            1.0
            if a1 and a1 == a2
            else 0.0
        ),

        "address_char_ratio": (
            ratio(a1, a2) / 100.0
            if a1 and a2
            else 0.0
        ),

        "address_jaro": (
            Jaro.normalized_similarity(
                a1,
                a2,
            )
            if a1 and a2
            else 0.0
        ),

        "address_jaro_winkler": (
            JaroWinkler.normalized_similarity(
                a1,
                a2,
            )
            if a1 and a2
            else 0.0
        ),

        "address_token_jaccard": (
            token_jaccard(
                a1,
                a2,
            )
        ),

        "address_token_overlap": (
            token_overlap(
                a1,
                a2,
            )
        ),

        "address_length_diff": (
            abs(
                len(a1) - len(a2)
            )
        ),

        "address_token_count_diff": (
            abs(
                len(a1.split())
                - len(a2.split())
            )
        ),

        "address_number_overlap": (
            number_overlap
        ),

        "address_number_match": (
            number_match
        ),

        "address_number_conflict": (
            number_conflict
        ),

        "address_postal_match": (
            postal_match
        ),

        "address_postal_conflict": (
            postal_conflict
        ),
    }


# =============================================================================
# COUNTRY FEATURES
# =============================================================================

def country_features(
    country1,
    country2,
):

    c1 = (
        str(country1)
        .strip()
        .upper()
        if not pd.isna(country1)
        else ""
    )

    c2 = (
        str(country2)
        .strip()
        .upper()
        if not pd.isna(country2)
        else ""
    )

    return {

        "country_exact": (
            1.0
            if c1 and c2 and c1 == c2
            else 0.0
        ),

        "country_missing": (
            1.0
            if not c1 or not c2
            else 0.0
        ),

        "country_conflict": (
            1.0
            if c1 and c2 and c1 != c2
            else 0.0
        ),
    }


# =============================================================================
# CONTRADICTION FEATURES
# =============================================================================

def contradiction_features(
    name_f,
    address_f,
    country_f,
):

    strong_name = (
        name_f["name_char_ratio"]
        >= 0.80
    )

    strong_address = (
        address_f["address_char_ratio"]
        >= 0.80
    )

    contradiction = (
        country_f["country_conflict"]
        == 1.0
        or
        address_f["address_number_conflict"]
        == 1.0
        or
        address_f["address_postal_conflict"]
        == 1.0
    )

    return {
        "strong_name_strong_address_conflict": (
            1.0
            if (
                strong_name
                and strong_address
                and contradiction
            )
            else 0.0
        )
    }


# =============================================================================
# STEP 1: COLLECT REQUIRED IDS
# =============================================================================

def collect_required_ids():

    print()
    print("=" * 80)
    print("STEP 1: COLLECT REQUIRED ENTITY IDS")
    print("=" * 80)

    s1_ids = set()
    candidate_ids = set()

    reader = pd.read_csv(
        PAIR_PATH,
        sep="\t",
        dtype=str,
        usecols=[
            "source1_entity_id",
            "candidate_entity_id",
        ],
        chunksize=PAIR_READ_CHUNK_SIZE,
        keep_default_na=False,
    )

    chunk_count = 0
    row_count = 0

    for chunk in reader:

        chunk_count += 1
        row_count += len(chunk)

        s1_ids.update(
            chunk[
                "source1_entity_id"
            ]
        )

        candidate_ids.update(
            chunk[
                "candidate_entity_id"
            ]
        )

        if chunk_count % 5 == 0:

            print(
                f"Scanned pair chunks: "
                f"{chunk_count:,} | "
                f"Rows: {row_count:,} | "
                f"S1 IDs: {len(s1_ids):,} | "
                f"Candidate IDs: {len(candidate_ids):,}"
            )

    print()
    print(
        f"Total pair rows: "
        f"{row_count:,}"
    )

    print(
        f"Unique S1 IDs required: "
        f"{len(s1_ids):,}"
    )

    print(
        f"Unique candidate IDs required: "
        f"{len(candidate_ids):,}"
    )

    return (
        s1_ids,
        candidate_ids,
    )


# =============================================================================
# STEP 2: BUILD SOURCE CACHE
# =============================================================================

def build_entity_cache(
    filename,
    required_ids,
    output_path,
):

    print()
    print(
        f"Building cache: {filename}"
    )

    print(
        f"Required IDs to look for: "
        f"{len(required_ids):,}"
    )

    source_path = (
        TRAIN_DIR / filename
    )

    if not source_path.exists():

        raise FileNotFoundError(
            f"Source file not found: "
            f"{source_path}"
        )

    usecols = [
        "entity_id",
        "business_name",
        "business_address",
        "country",
    ]

    found = 0
    chunks = 0

    first_write = True

    # IMPORTANT:
    # Scan the ENTIRE source file.
    #
    # We cannot stop after an arbitrary number of chunks because
    # required IDs can appear anywhere in the source file.
    for chunk in pd.read_csv(
        source_path,
        sep="\t",
        usecols=usecols,
        dtype=str,
        chunksize=ENTITY_SCAN_CHUNK_SIZE,
        keep_default_na=False,
    ):

        chunks += 1

        mask = chunk[
            "entity_id"
        ].isin(required_ids)

        if mask.any():

            selected = chunk.loc[mask]

            selected.to_csv(
                output_path,
                sep="\t",
                index=False,
                mode=(
                    "w"
                    if first_write
                    else "a"
                ),
                header=first_write,
            )

            first_write = False

            found += len(selected)

        if chunks % 5 == 0:

            print(
                f"Source chunks scanned: "
                f"{chunks:,} | "
                f"Entities cached: "
                f"{found:,}"
            )

    print()
    print(
        f"Finished {filename}"
    )

    print(
        f"Total source chunks scanned: "
        f"{chunks:,}"
    )

    print(
        f"Entities cached: "
        f"{found:,}"
    )

    return found


# =============================================================================
# LOAD CACHE
# =============================================================================

def load_cache(path):

    print(
        f"Loading cache: "
        f"{path.name}"
    )

    df = pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
    )

    lookup = {}

    for row in df.itertuples(
        index=False
    ):

        lookup[row.entity_id] = (
            row.business_name,
            row.business_address,
            row.country,
        )

    count = len(lookup)

    del df

    gc.collect()

    print(
        f"Loaded entities: "
        f"{count:,}"
    )

    return lookup


# =============================================================================
# STEP 3: GENERATE FEATURES
# =============================================================================

def generate_features(
    s1_lookup,
    candidate_lookup,
):

    print()
    print("=" * 80)
    print("STEP 3: GENERATE 28 PAIR FEATURES")
    print("=" * 80)

    if OUTPUT_PATH.exists():

        print(
            "Removing previous feature output:"
        )

        print(
            OUTPUT_PATH
        )

        OUTPUT_PATH.unlink()

    first_write = True

    total_rows = 0
    chunk_number = 0

    reader = pd.read_csv(
        PAIR_PATH,
        sep="\t",
        dtype={
            "source1_entity_id": str,
            "candidate_entity_id": str,
            "label": "int8",
        },
        chunksize=PAIR_READ_CHUNK_SIZE,
        keep_default_na=False,
    )

    for pairs in reader:

        chunk_number += 1

        print()
        print("-" * 80)

        print(
            f"Processing chunk "
            f"{chunk_number:,}"
        )

        print(
            f"Rows in chunk: "
            f"{len(pairs):,}"
        )

        rows = []

        missing_count = 0

        for pair in pairs.itertuples(
            index=False
        ):

            s1 = s1_lookup.get(
                pair.source1_entity_id
            )

            candidate = candidate_lookup.get(
                pair.candidate_entity_id
            )

            if (
                s1 is None
                or candidate is None
            ):

                missing_count += 1

                continue

            (
                s1_name,
                s1_address,
                s1_country,
            ) = s1

            (
                candidate_name,
                candidate_address,
                candidate_country,
            ) = candidate

            # -------------------------------------------------------------
            # Name
            # -------------------------------------------------------------

            name_f = name_features(
                s1_name,
                candidate_name,
            )

            # -------------------------------------------------------------
            # Address
            # -------------------------------------------------------------

            address_f = address_features(
                s1_address,
                candidate_address,
            )

            # -------------------------------------------------------------
            # Country
            # -------------------------------------------------------------

            country_f = country_features(
                s1_country,
                candidate_country,
            )

            # -------------------------------------------------------------
            # Contradiction
            # -------------------------------------------------------------

            contradiction_f = (
                contradiction_features(
                    name_f,
                    address_f,
                    country_f,
                )
            )

            # -------------------------------------------------------------
            # Combine
            # -------------------------------------------------------------

            row = {

                "source1_entity_id":
                    pair.source1_entity_id,

                "candidate_entity_id":
                    pair.candidate_entity_id,

                "label":
                    int(pair.label),
            }

            row.update(name_f)
            row.update(address_f)
            row.update(country_f)
            row.update(contradiction_f)

            rows.append(row)

        features = pd.DataFrame(
            rows
        )

        features.to_csv(
            OUTPUT_PATH,
            sep="\t",
            index=False,
            mode=(
                "w"
                if first_write
                else "a"
            ),
            header=first_write,
        )

        first_write = False

        total_rows += len(features)

        print(
            f"Rows written: "
            f"{total_rows:,}"
        )

        if missing_count > 0:

            print(
                f"WARNING: "
                f"{missing_count:,} pairs "
                f"could not be feature-engineered."
            )

        del pairs
        del features
        del rows

        gc.collect()

    print()
    print("=" * 80)
    print("FEATURE GENERATION COMPLETE")
    print("=" * 80)

    print(
        f"Chunks processed: "
        f"{chunk_number:,}"
    )

    print(
        f"Rows written: "
        f"{total_rows:,}"
    )

    print()
    print(
        f"Output:"
    )

    print(
        OUTPUT_PATH
    )

    return total_rows


# =============================================================================
# MAIN
# =============================================================================

def main():

    print("=" * 80)
    print("OPTIMIZED PAIR FEATURE PIPELINE")
    print("=" * 80)

    # -------------------------------------------------------------------------
    # Validate pair file
    # -------------------------------------------------------------------------

    if not PAIR_PATH.exists():

        print()
        print(
            "ERROR: Pair file not found:"
        )

        print(
            PAIR_PATH
        )

        sys.exit(1)

    # -------------------------------------------------------------------------
    # Validate source files
    # -------------------------------------------------------------------------

    source_files = [
        "train_source1.tsv",
        "train_source2.tsv",
        "train_source3.tsv",
    ]

    for filename in source_files:

        path = (
            TRAIN_DIR / filename
        )

        if not path.exists():

            print()
            print(
                "ERROR: Source file not found:"
            )

            print(
                path
            )

            sys.exit(1)

    # -------------------------------------------------------------------------
    # Create cache directory
    # -------------------------------------------------------------------------

    CACHE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # -------------------------------------------------------------------------
    # Remove stale caches
    # -------------------------------------------------------------------------

    cache_paths = {

        "s1": (
            CACHE_DIR
            / "required_s1.tsv"
        ),

        "s2": (
            CACHE_DIR
            / "required_s2.tsv"
        ),

        "s3": (
            CACHE_DIR
            / "required_s3.tsv"
        ),
    }

    for path in cache_paths.values():

        if path.exists():

            print(
                f"Removing old cache: "
                f"{path.name}"
            )

            path.unlink()

    # -------------------------------------------------------------------------
    # STEP 1
    # -------------------------------------------------------------------------

    (
        s1_ids,
        candidate_ids,
    ) = collect_required_ids()

    # -------------------------------------------------------------------------
    # STEP 2A - S1
    # -------------------------------------------------------------------------

    s1_found = build_entity_cache(
        "train_source1.tsv",
        s1_ids,
        cache_paths["s1"],
    )

    # -------------------------------------------------------------------------
    # STEP 2B - S2
    #
    # candidate_ids contains both S2 and S3 IDs.
    # It is intentional that S2 finds only its own subset.
    # -------------------------------------------------------------------------

    s2_found = build_entity_cache(
        "train_source2.tsv",
        candidate_ids,
        cache_paths["s2"],
    )

    # -------------------------------------------------------------------------
    # STEP 2C - S3
    # -------------------------------------------------------------------------

    s3_found = build_entity_cache(
        "train_source3.tsv",
        candidate_ids,
        cache_paths["s3"],
    )

    # -------------------------------------------------------------------------
    # CACHE INTEGRITY CHECK
    # -------------------------------------------------------------------------

    print()
    print("=" * 80)
    print("CACHE INTEGRITY CHECK")
    print("=" * 80)

    print(
        f"Required S1 IDs: "
        f"{len(s1_ids):,}"
    )

    print(
        f"S1 cached: "
        f"{s1_found:,}"
    )

    print()

    print(
        f"Required candidate IDs: "
        f"{len(candidate_ids):,}"
    )

    print(
        f"S2 cached: "
        f"{s2_found:,}"
    )

    print(
        f"S3 cached: "
        f"{s3_found:,}"
    )

    print(
        f"S2 + S3 cached: "
        f"{s2_found + s3_found:,}"
    )

    # -------------------------------------------------------------------------
    # S1 must match exactly.
    # -------------------------------------------------------------------------

    if s1_found != len(s1_ids):

        print()
        print(
            "ERROR: Not all required S1 entities "
            "were found."
        )

        print(
            f"Missing: "
            f"{len(s1_ids) - s1_found:,}"
        )

        sys.exit(1)

    # -------------------------------------------------------------------------
    # S2 + S3 together must cover all candidates.
    # -------------------------------------------------------------------------

    if (
        s2_found + s3_found
        != len(candidate_ids)
    ):

        print()
        print(
            "ERROR: S2 + S3 cache count "
            "does not equal required candidate count."
        )

        print(
            f"Required: "
            f"{len(candidate_ids):,}"
        )

        print(
            f"Found: "
            f"{s2_found + s3_found:,}"
        )

        sys.exit(1)

    print()
    print(
        "CACHE INTEGRITY: PASS"
    )

    # -------------------------------------------------------------------------
    # Load caches
    # -------------------------------------------------------------------------

    print()
    print("=" * 80)
    print("LOADING ENTITY CACHES")
    print("=" * 80)

    s1_lookup = load_cache(
        cache_paths["s1"]
    )

    s2_lookup = load_cache(
        cache_paths["s2"]
    )

    s3_lookup = load_cache(
        cache_paths["s3"]
    )

    # -------------------------------------------------------------------------
    # Combine S2 + S3.
    #
    # entity_id prefixes make source IDs distinct.
    # -------------------------------------------------------------------------

    candidate_lookup = {}

    candidate_lookup.update(
        s2_lookup
    )

    candidate_lookup.update(
        s3_lookup
    )

    del s2_lookup
    del s3_lookup

    gc.collect()

    print()
    print(
        f"Combined candidate entities: "
        f"{len(candidate_lookup):,}"
    )

    if (
        len(candidate_lookup)
        != len(candidate_ids)
    ):

        print()
        print(
            "ERROR: Combined candidate lookup "
            "does not contain all required IDs."
        )

        sys.exit(1)

    # -------------------------------------------------------------------------
    # STEP 3
    # -------------------------------------------------------------------------

    total_rows = generate_features(
        s1_lookup,
        candidate_lookup,
    )

    # -------------------------------------------------------------------------
    # FINAL CHECK
    # -------------------------------------------------------------------------

    print()
    print("=" * 80)
    print("FINAL CHECK")
    print("=" * 80)

    expected_rows = 2_000_000

    print(
        f"Expected rows: "
        f"{expected_rows:,}"
    )

    print(
        f"Generated rows: "
        f"{total_rows:,}"
    )

    if total_rows == expected_rows:

        print()
        print(
            "STATUS: PASS"
        )

    else:

        print()
        print(
            "STATUS: WARNING"
        )

        print(
            "The generated row count "
            "does not equal 2,000,000."
        )

    print()
    print(
        "Feature output:"
    )

    print(
        OUTPUT_PATH
    )


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    main()
import os
import re
import pickle
import math
from pathlib import Path

import pandas as pd
from rapidfuzz.distance import Levenshtein, Jaro, JaroWinkler


# =============================================================================
# STAGE 2 - VALIDATION PAIR FEATURE GENERATION
# =============================================================================

BASE_DIR = Path(__file__).resolve().parent.parent

PAIR_PATH = BASE_DIR / "experiments" / "004_ml_baseline" / "dev_candidate_pairs.tsv"
OUTPUT_PATH = BASE_DIR / "experiments" / "004_ml_baseline" / "validation_pair_features.tsv"

CACHE_DIR = (
    BASE_DIR
    / "experiments"
    / "004_ml_baseline"
    / "validation_entity_cache"
)

DATASET_DIR = BASE_DIR / "dataset" / "train"

S1_PATH = DATASET_DIR / "train_source1.tsv"
S2_PATH = DATASET_DIR / "train_source2.tsv"
S3_PATH = DATASET_DIR / "train_source3.tsv"

PAIR_READ_CHUNK_SIZE = 50_000
ENTITY_SCAN_CHUNK_SIZE = 250_000


# =============================================================================
# TEXT NORMALIZATION
# =============================================================================

def normalize_text(value):
    if pd.isna(value):
        return ""

    value = str(value).lower().strip()

    # Replace punctuation/symbols with spaces.
    value = re.sub(r"[^a-z0-9]+", " ", value)

    # Collapse whitespace.
    value = re.sub(r"\s+", " ", value).strip()

    return value


def compact_text(value):
    return re.sub(r"[^a-z0-9]", "", normalize_text(value))


def tokenize(value):
    text = normalize_text(value)

    if not text:
        return []

    return text.split()


def sorted_tokens_text(value):
    tokens = tokenize(value)

    if not tokens:
        return ""

    return " ".join(sorted(tokens))


def remove_legal_suffixes(value):
    tokens = tokenize(value)

    if not tokens:
        return ""

    suffixes = {
        "inc",
        "incorporated",
        "corp",
        "corporation",
        "co",
        "company",
        "llc",
        "ltd",
        "limited",
        "plc",
        "pvt",
        "private",
        "llp",
        "group",
        "holdings",
        "enterprises",
        "enterprise",
        "industries",
        "international",
        "intl",
    }

    filtered = [token for token in tokens if token not in suffixes]

    return " ".join(filtered)


def extract_numbers(value):
    text = normalize_text(value)

    if not text:
        return set()

    return set(re.findall(r"\d+", text))


# =============================================================================
# SIMILARITY HELPERS
# =============================================================================

def safe_ratio(a, b):
    if not a and not b:
        return 1.0

    if not a or not b:
        return 0.0

    return Levenshtein.normalized_similarity(a, b)


def jaro_similarity(a, b):
    if not a and not b:
        return 1.0

    if not a or not b:
        return 0.0

    return Jaro.normalized_similarity(a, b)


def jaro_winkler_similarity(a, b):
    if not a and not b:
        return 1.0

    if not a or not b:
        return 0.0

    return JaroWinkler.normalized_similarity(a, b)


def token_jaccard(tokens_a, tokens_b):
    a = set(tokens_a)
    b = set(tokens_b)

    if not a and not b:
        return 1.0

    union = a | b

    if not union:
        return 0.0

    return len(a & b) / len(union)


def token_overlap(tokens_a, tokens_b):
    a = set(tokens_a)
    b = set(tokens_b)

    if not a or not b:
        return 0.0

    return len(a & b) / min(len(a), len(b))


def number_overlap(numbers_a, numbers_b):
    if not numbers_a or not numbers_b:
        return 0.0

    return len(numbers_a & numbers_b) / min(
        len(numbers_a),
        len(numbers_b)
    )


def number_match(numbers_a, numbers_b):
    if not numbers_a or not numbers_b:
        return 0

    return int(bool(numbers_a & numbers_b))


def number_conflict(numbers_a, numbers_b):
    if not numbers_a or not numbers_b:
        return 0

    return int(not bool(numbers_a & numbers_b))


def extract_postal(value):
    text = normalize_text(value)

    if not text:
        return ""

    # Generic numeric postal-code extraction.
    candidates = re.findall(r"\b\d{4,6}\b", text)

    if not candidates:
        return ""

    return candidates[0]


# =============================================================================
# ENTITY PREPARATION
# =============================================================================

def prepare_entity(row):
    business_name = row.get("business_name", "")
    business_address = row.get("business_address", "")
    country = row.get("country", "")

    name = normalize_text(business_name)
    address = normalize_text(business_address)
    country_norm = normalize_text(country)

    name_tokens = tokenize(name)
    address_tokens = tokenize(address)

    return {
        "business_name": business_name,
        "business_address": business_address,
        "country": country,

        "name": name,
        "address": address,
        "country_norm": country_norm,

        "name_tokens": name_tokens,
        "address_tokens": address_tokens,

        "name_no_suffix": remove_legal_suffixes(name),
        "name_compact": compact_text(name),
        "name_sorted": sorted_tokens_text(name),

        "name_length": len(name),
        "address_length": len(address),

        "name_token_count": len(name_tokens),
        "address_token_count": len(address_tokens),

        "address_numbers": extract_numbers(address),
        "postal": extract_postal(address),
    }


# =============================================================================
# CACHE UTILITIES
# =============================================================================

def cache_path(source_name):
    return CACHE_DIR / f"{source_name}_cache.pkl"


def save_cache(data, source_name):
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    path = cache_path(source_name)

    with open(path, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_cache(source_name):
    path = cache_path(source_name)

    if not path.exists():
        return None

    print(f"Loading existing {source_name} cache: {path}")

    with open(path, "rb") as f:
        data = pickle.load(f)

    return data


# =============================================================================
# ENTITY CACHE BUILDING
# =============================================================================

def collect_required_ids():
    print("Collecting required entity IDs...")

    required_s1 = set()
    required_candidates = set()

    for chunk in pd.read_csv(
        PAIR_PATH,
        sep="\t",
        dtype=str,
        usecols=[
            "source1_entity_id",
            "candidate_entity_id",
        ],
        chunksize=PAIR_READ_CHUNK_SIZE,
    ):
        required_s1.update(chunk["source1_entity_id"].dropna())
        required_candidates.update(chunk["candidate_entity_id"].dropna())

    print(f"Required S1 IDs: {len(required_s1):,}")
    print(f"Required candidate IDs: {len(required_candidates):,}")

    return required_s1, required_candidates


def build_entity_cache(
    source_path,
    source_name,
    required_ids,
):
    existing = load_cache(source_name)

    if existing is not None:

        # Verify cache contains all requested IDs.
        missing = required_ids.difference(existing.keys())

        if not missing:
            print(
                f"{source_name} cache already complete: "
                f"{len(existing):,}"
            )
            return existing

        print(
            f"{source_name} cache exists but is missing "
            f"{len(missing):,} required IDs."
        )

    print()
    print(f"Building {source_name} cache...")

    cache = {} if existing is None else existing

    usecols = [
        "entity_id",
        "business_name",
        "business_address",
        "country",
    ]

    processed = 0

    for chunk in pd.read_csv(
        source_path,
        sep="\t",
        dtype=str,
        usecols=usecols,
        chunksize=ENTITY_SCAN_CHUNK_SIZE,
    ):

        for row in chunk.itertuples(index=False):

            entity_id = row.entity_id

            if entity_id in required_ids and entity_id not in cache:

                row_dict = {
                    "entity_id": entity_id,
                    "business_name": row.business_name,
                    "business_address": row.business_address,
                    "country": row.country,
                }

                cache[entity_id] = prepare_entity(row_dict)

        processed += len(chunk)

        if len(cache) % 100_000 < len(chunk):
            print(
                f"{source_name} cached: "
                f"{len(cache):,}"
            )

        # Stop once every required ID is available.
        if len(cache) >= len(required_ids):
            break

    missing = required_ids.difference(cache.keys())

    if missing:
        print(
            f"ERROR: {source_name} cache missing "
            f"{len(missing):,} IDs."
        )

        sample = list(missing)[:10]

        print("Example missing IDs:")
        for x in sample:
            print(x)

        raise RuntimeError(
            f"{source_name} cache incomplete."
        )

    save_cache(cache, source_name)

    print(
        f"{source_name} cache complete: "
        f"{len(cache):,}"
    )

    return cache


# =============================================================================
# FEATURE GENERATION
# =============================================================================

def build_feature_row(
    s1_id,
    candidate_id,
    s1_row,
    candidate_row,
    label,
):
    """
    IMPORTANT:
    Entity IDs are supplied separately.

    Do NOT use:
        s1_row["entity_id"]
        candidate_row["entity_id"]

    because the cached feature dictionaries do not depend on
    an entity_id field being present.
    """

    # -------------------------------------------------------------------------
    # NAME
    # -------------------------------------------------------------------------

    name_a = s1_row["name"]
    name_b = candidate_row["name"]

    name_tokens_a = s1_row["name_tokens"]
    name_tokens_b = candidate_row["name_tokens"]

    name_no_suffix_a = s1_row["name_no_suffix"]
    name_no_suffix_b = candidate_row["name_no_suffix"]

    name_compact_a = s1_row["name_compact"]
    name_compact_b = candidate_row["name_compact"]

    name_sorted_a = s1_row["name_sorted"]
    name_sorted_b = candidate_row["name_sorted"]

    # -------------------------------------------------------------------------
    # ADDRESS
    # -------------------------------------------------------------------------

    address_a = s1_row["address"]
    address_b = candidate_row["address"]

    address_tokens_a = s1_row["address_tokens"]
    address_tokens_b = candidate_row["address_tokens"]

    numbers_a = s1_row["address_numbers"]
    numbers_b = candidate_row["address_numbers"]

    postal_a = s1_row["postal"]
    postal_b = candidate_row["postal"]

    # -------------------------------------------------------------------------
    # COUNTRY
    # -------------------------------------------------------------------------

    country_a = s1_row["country_norm"]
    country_b = candidate_row["country_norm"]

    country_missing = int(
        not country_a or not country_b
    )

    country_exact = int(
        bool(country_a)
        and bool(country_b)
        and country_a == country_b
    )

    country_conflict = int(
        bool(country_a)
        and bool(country_b)
        and country_a != country_b
    )

    # -------------------------------------------------------------------------
    # NAME FEATURES
    # -------------------------------------------------------------------------

    name_exact = int(
        bool(name_a)
        and bool(name_b)
        and name_a == name_b
    )

    name_char_ratio = safe_ratio(
        name_a,
        name_b,
    )

    name_jaro = jaro_similarity(
        name_a,
        name_b,
    )

    name_jaro_winkler = jaro_winkler_similarity(
        name_a,
        name_b,
    )

    name_token_jaccard = token_jaccard(
        name_tokens_a,
        name_tokens_b,
    )

    name_token_overlap = token_overlap(
        name_tokens_a,
        name_tokens_b,
    )

    name_length_diff = abs(
        s1_row["name_length"]
        - candidate_row["name_length"]
    )

    name_token_count_diff = abs(
        s1_row["name_token_count"]
        - candidate_row["name_token_count"]
    )

    name_no_suffix_ratio = safe_ratio(
        name_no_suffix_a,
        name_no_suffix_b,
    )

    name_compact_ratio = safe_ratio(
        name_compact_a,
        name_compact_b,
    )

    name_sorted_ratio = safe_ratio(
        name_sorted_a,
        name_sorted_b,
    )

    # -------------------------------------------------------------------------
    # ADDRESS FEATURES
    # -------------------------------------------------------------------------

    address_exact = int(
        bool(address_a)
        and bool(address_b)
        and address_a == address_b
    )

    address_char_ratio = safe_ratio(
        address_a,
        address_b,
    )

    address_jaro = jaro_similarity(
        address_a,
        address_b,
    )

    address_jaro_winkler = jaro_winkler_similarity(
        address_a,
        address_b,
    )

    address_token_jaccard = token_jaccard(
        address_tokens_a,
        address_tokens_b,
    )

    address_token_overlap = token_overlap(
        address_tokens_a,
        address_tokens_b,
    )

    address_length_diff = abs(
        s1_row["address_length"]
        - candidate_row["address_length"]
    )

    address_token_count_diff = abs(
        s1_row["address_token_count"]
        - candidate_row["address_token_count"]
    )

    address_number_overlap = number_overlap(
        numbers_a,
        numbers_b,
    )

    address_number_match = number_match(
        numbers_a,
        numbers_b,
    )

    address_number_conflict = number_conflict(
        numbers_a,
        numbers_b,
    )

    address_postal_match = int(
        bool(postal_a)
        and bool(postal_b)
        and postal_a == postal_b
    )

    address_postal_conflict = int(
        bool(postal_a)
        and bool(postal_b)
        and postal_a != postal_b
    )

    # -------------------------------------------------------------------------
    # CONTRADICTION FEATURE
    # -------------------------------------------------------------------------

    strong_name = (
        name_char_ratio >= 0.80
        or name_jaro_winkler >= 0.90
        or name_token_overlap >= 0.80
    )

    strong_address = (
        address_char_ratio >= 0.80
        or address_jaro_winkler >= 0.90
        or address_token_overlap >= 0.80
    )

    strong_name_strong_address_conflict = int(
        strong_name
        and strong_address
        and (
            country_conflict
            or address_number_conflict
            or address_postal_conflict
        )
    )

    # -------------------------------------------------------------------------
    # FINAL FEATURE ROW
    # -------------------------------------------------------------------------

    return {
        "source1_entity_id": s1_id,
        "candidate_entity_id": candidate_id,

        # NAME - 11
        "name_exact": name_exact,
        "name_char_ratio": name_char_ratio,
        "name_jaro": name_jaro,
        "name_jaro_winkler": name_jaro_winkler,
        "name_token_jaccard": name_token_jaccard,
        "name_token_overlap": name_token_overlap,
        "name_length_diff": name_length_diff,
        "name_token_count_diff": name_token_count_diff,
        "name_no_suffix_ratio": name_no_suffix_ratio,
        "name_compact_ratio": name_compact_ratio,
        "name_sorted_ratio": name_sorted_ratio,

        # ADDRESS - 13
        "address_exact": address_exact,
        "address_char_ratio": address_char_ratio,
        "address_jaro": address_jaro,
        "address_jaro_winkler": address_jaro_winkler,
        "address_token_jaccard": address_token_jaccard,
        "address_token_overlap": address_token_overlap,
        "address_length_diff": address_length_diff,
        "address_token_count_diff": address_token_count_diff,
        "address_number_overlap": address_number_overlap,
        "address_number_match": address_number_match,
        "address_number_conflict": address_number_conflict,
        "address_postal_match": address_postal_match,
        "address_postal_conflict": address_postal_conflict,

        # COUNTRY - 3
        "country_exact": country_exact,
        "country_missing": country_missing,
        "country_conflict": country_conflict,

        # CONTRADICTION - 1
        "strong_name_strong_address_conflict":
            strong_name_strong_address_conflict,

        # LABEL
        "label": int(label),
    }


# =============================================================================
# MAIN
# =============================================================================

def main():

    print("=" * 80)
    print("STAGE 2 - VALIDATION PAIR FEATURE GENERATION")
    print("=" * 80)

    if not PAIR_PATH.exists():
        raise FileNotFoundError(
            f"Candidate pair file not found:\n{PAIR_PATH}"
        )

    CACHE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # -------------------------------------------------------------------------
    # 1. Collect IDs
    # -------------------------------------------------------------------------

    required_s1, required_candidates = collect_required_ids()

    # Candidate IDs are globally unique because S2/S3 IDs have different
    # prefixes.
    required_s2 = {
        x for x in required_candidates
        if str(x).startswith("S2-")
    }

    required_s3 = {
        x for x in required_candidates
        if str(x).startswith("S3-")
    }

    print()
    print(f"Required S2 IDs: {len(required_s2):,}")
    print(f"Required S3 IDs: {len(required_s3):,}")

    # -------------------------------------------------------------------------
    # 2. Load/build caches
    # -------------------------------------------------------------------------

    s1_cache = build_entity_cache(
        S1_PATH,
        "s1",
        required_s1,
    )

    s2_cache = build_entity_cache(
        S2_PATH,
        "s2",
        required_s2,
    )

    s3_cache = build_entity_cache(
        S3_PATH,
        "s3",
        required_s3,
    )

    # -------------------------------------------------------------------------
    # 3. Cache integrity
    # -------------------------------------------------------------------------

    missing_s1 = required_s1.difference(s1_cache.keys())
    missing_s2 = required_s2.difference(s2_cache.keys())
    missing_s3 = required_s3.difference(s3_cache.keys())

    print()
    print("CACHE INTEGRITY CHECK")
    print("-" * 80)

    print(
        f"S1 required : {len(required_s1):,}"
    )
    print(
        f"S1 cached   : {len(required_s1) - len(missing_s1):,}"
    )

    print(
        f"S2 required : {len(required_s2):,}"
    )
    print(
        f"S2 cached   : {len(required_s2) - len(missing_s2):,}"
    )

    print(
        f"S3 required : {len(required_s3):,}"
    )
    print(
        f"S3 cached   : {len(required_s3) - len(missing_s3):,}"
    )

    if missing_s1 or missing_s2 or missing_s3:
        raise RuntimeError(
            "CACHE INTEGRITY FAILED."
        )

    print("CACHE INTEGRITY: PASS")

    # -------------------------------------------------------------------------
    # 4. Combine candidate cache
    # -------------------------------------------------------------------------

    print()
    print("Loading entity caches...")

    candidate_cache = {}

    candidate_cache.update(s2_cache)
    candidate_cache.update(s3_cache)

    print(
        f"S1 entities loaded: {len(s1_cache):,}"
    )

    print(
        f"Candidate entities loaded: "
        f"{len(candidate_cache):,}"
    )

    # -------------------------------------------------------------------------
    # 5. Remove previous output
    # -------------------------------------------------------------------------

    if OUTPUT_PATH.exists():
        print()
        print(
            f"Removing previous output:\n{OUTPUT_PATH}"
        )

        OUTPUT_PATH.unlink()

    # -------------------------------------------------------------------------
    # 6. Feature generation
    # -------------------------------------------------------------------------

    print()
    print("Generating features...")

    first_write = True

    total_rows = 0
    chunk_number = 0

    feature_columns = [
        "source1_entity_id",
        "candidate_entity_id",

        # Name
        "name_exact",
        "name_char_ratio",
        "name_jaro",
        "name_jaro_winkler",
        "name_token_jaccard",
        "name_token_overlap",
        "name_length_diff",
        "name_token_count_diff",
        "name_no_suffix_ratio",
        "name_compact_ratio",
        "name_sorted_ratio",

        # Address
        "address_exact",
        "address_char_ratio",
        "address_jaro",
        "address_jaro_winkler",
        "address_token_jaccard",
        "address_token_overlap",
        "address_length_diff",
        "address_token_count_diff",
        "address_number_overlap",
        "address_number_match",
        "address_number_conflict",
        "address_postal_match",
        "address_postal_conflict",

        # Country
        "country_exact",
        "country_missing",
        "country_conflict",

        # Contradiction
        "strong_name_strong_address_conflict",

        # Label
        "label",
    ]

    for chunk in pd.read_csv(
        PAIR_PATH,
        sep="\t",
        dtype=str,
        chunksize=PAIR_READ_CHUNK_SIZE,
    ):

        chunk_number += 1

        print(
            f"Processing chunk {chunk_number}"
        )

        rows = []

        for pair in chunk.itertuples(index=False):

            s1_id = pair.source1_entity_id
            candidate_id = pair.candidate_entity_id

            label = int(pair.label)

            s1_row = s1_cache.get(s1_id)
            candidate_row = candidate_cache.get(candidate_id)

            if s1_row is None:
                raise RuntimeError(
                    f"Missing S1 entity from cache: {s1_id}"
                )

            if candidate_row is None:
                raise RuntimeError(
                    f"Missing candidate entity from cache: "
                    f"{candidate_id}"
                )

            feature_row = build_feature_row(
                s1_id,
                candidate_id,
                s1_row,
                candidate_row,
                label,
            )

            rows.append(feature_row)

        feature_df = pd.DataFrame(
            rows,
            columns=feature_columns,
        )

        feature_df.to_csv(
            OUTPUT_PATH,
            sep="\t",
            index=False,
            mode="w" if first_write else "a",
            header=first_write,
        )

        first_write = False

        total_rows += len(feature_df)

        print(
            f"Rows written so far: "
            f"{total_rows:,}"
        )

    # -------------------------------------------------------------------------
    # 7. Final validation
    # -------------------------------------------------------------------------

    print()
    print("=" * 80)
    print("STAGE 2 VALIDATION")
    print("=" * 80)

    if not OUTPUT_PATH.exists():
        raise RuntimeError(
            "Output file was not created."
        )

    output_size = OUTPUT_PATH.stat().st_size

    print(
        f"Output file: {OUTPUT_PATH}"
    )

    print(
        f"Output size: "
        f"{output_size / (1024 * 1024):.2f} MB"
    )

    print(
        f"Rows written: "
        f"{total_rows:,}"
    )

    if total_rows == 0:
        raise RuntimeError(
            "Zero feature rows generated."
        )

    # Read only header + small sample for schema verification.
    check = pd.read_csv(
        OUTPUT_PATH,
        sep="\t",
        nrows=5,
    )

    missing_columns = [
        c for c in feature_columns
        if c not in check.columns
    ]

    if missing_columns:
        raise RuntimeError(
            f"Missing output columns: {missing_columns}"
        )

    expected_feature_count = 28

    actual_feature_count = len(
        [
            c for c in feature_columns
            if c not in {
                "source1_entity_id",
                "candidate_entity_id",
                "label",
            }
        ]
    )

    print(
        f"Feature columns: "
        f"{actual_feature_count}"
    )

    if actual_feature_count != expected_feature_count:
        raise RuntimeError(
            f"Expected {expected_feature_count} "
            f"features but found "
            f"{actual_feature_count}."
        )

    print()
    print("STATUS: PASS")
    print("=" * 80)


if __name__ == "__main__":
    main()
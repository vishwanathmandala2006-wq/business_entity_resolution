from pathlib import Path
from collections import defaultdict, Counter
import re
import unicodedata
import pandas as pd
import time


# ============================================================================
# CONFIG
# ============================================================================

BASE_DIR = Path(__file__).resolve().parent.parent

TRAIN_DIR = BASE_DIR / "dataset" / "train"
EXP_DIR = BASE_DIR / "experiments" / "001_baseline"

S1_PATH = TRAIN_DIR / "train_source1.tsv"
S2_PATH = TRAIN_DIR / "train_source2.tsv"
S3_PATH = TRAIN_DIR / "train_source3.tsv"

GT_PATH = EXP_DIR / "validation_ground_truth.tsv"

EXISTING_CANDIDATES = EXP_DIR / "structural_candidates.tsv"
OUTPUT_PATH = EXP_DIR / "approx_candidates.tsv"

MAX_KEY_FREQUENCY = 100
MIN_TOKEN_LENGTH = 4

CHUNK_SIZE = 250_000


# ============================================================================
# NORMALIZATION
# ============================================================================

def normalize_unicode(value):
    """
    Unicode-aware normalization.

    Keeps non-Latin characters instead of deleting them.
    """
    if pd.isna(value):
        return ""

    value = str(value).lower().strip()

    value = unicodedata.normalize("NFKC", value)

    value = re.sub(r"[^\w\s]", " ", value, flags=re.UNICODE)

    value = re.sub(r"\s+", " ", value)

    return value.strip()


def ascii_fold(value):
    """
    Secondary representation useful for transliteration-like
    Latin variations.
    """
    if not value:
        return ""

    value = unicodedata.normalize("NFKD", value)

    return "".join(
        c for c in value
        if not unicodedata.combining(c)
    )


def tokens(value):
    value = normalize_unicode(value)

    if not value:
        return []

    return [
        x for x in value.split()
        if len(x) >= MIN_TOKEN_LENGTH
    ]


def numbers(value):
    if pd.isna(value):
        return []

    return re.findall(r"\d+", str(value))


# ============================================================================
# CHARACTER NGRAMS
# ============================================================================

def char_ngrams(value, n=3):

    value = normalize_unicode(value)

    compact = value.replace(" ", "")

    if len(compact) < n:
        return []

    return [
        compact[i:i+n]
        for i in range(len(compact) - n + 1)
    ]


# ============================================================================
# ADDRESS SIGNATURES
# ============================================================================

def address_signatures(address, country):

    result = []

    norm = normalize_unicode(address)

    if not norm:
        return result

    nums = numbers(address)
    toks = tokens(address)

    # House-number + informative token
    if nums:
        house = nums[0]

        for token in toks:
            if len(token) >= 5:
                result.append(
                    ("ADDR_NUM_TOKEN", house, token)
                )

    # Two informative address tokens
    informative = [
        x for x in toks
        if len(x) >= 6
    ]

    if len(informative) >= 2:
        for i in range(min(len(informative), 8)):
            for j in range(i + 1, min(len(informative), 8)):
                result.append(
                    (
                        "ADDR_TOKEN_PAIR",
                        informative[i],
                        informative[j]
                    )
                )

    # Number + country
    if nums and country:
        result.append(
            (
                "ADDR_NUM_COUNTRY",
                nums[0],
                normalize_unicode(country)
            )
        )

    return result


# ============================================================================
# NAME SIGNATURES
# ============================================================================

def name_signatures(name, country):

    result = []

    norm = normalize_unicode(name)

    if not norm:
        return result

    toks = tokens(name)

    # Rare individual token
    for token in toks:
        if len(token) >= 5:
            result.append(
                ("NAME_TOKEN", token)
            )

    # Two-token combinations
    informative = [
        x for x in toks
        if len(x) >= 5
    ]

    if len(informative) >= 2:
        # Preserve order
        for i in range(min(len(informative), 6)):
            for j in range(i + 1, min(len(informative), 6)):
                result.append(
                    (
                        "NAME_TOKEN_PAIR",
                        informative[i],
                        informative[j]
                    )
                )

    # Character trigram signatures
    grams = char_ngrams(name, 3)

    if grams:
        counts = Counter(grams)

        # Only use distinctive-looking grams.
        for gram, count in counts.items():
            if count == 1:
                result.append(
                    ("NAME_3GRAM", gram)
                )

    # ASCII-folded representation
    folded = ascii_fold(norm)

    if folded and folded != norm:
        folded_grams = char_ngrams(folded, 3)

        for gram in set(folded_grams):
            result.append(
                ("NAME_FOLD_3GRAM", gram)
            )

    return result


# ============================================================================
# LOAD VALIDATION S1
# ============================================================================

def load_validation_s1():

    gt = pd.read_csv(
        GT_PATH,
        sep="\t",
        dtype=str,
        keep_default_na=False
    )

    ids = set(gt["source1_entity_id"])

    s1 = pd.read_csv(
        S1_PATH,
        sep="\t",
        dtype=str,
        keep_default_na=False
    )

    s1 = s1[s1["entity_id"].isin(ids)].copy()

    return s1


# ============================================================================
# BUILD QUERY SIGNATURES
# ============================================================================

def build_query_signatures(s1):

    name_keys = set()
    address_keys = set()

    for row in s1.itertuples(index=False):

        for key in name_signatures(
            row.business_name,
            row.country
        ):
            name_keys.add(key)

        for key in address_signatures(
            row.business_address,
            row.country
        ):
            address_keys.add(key)

    return name_keys, address_keys


# ============================================================================
# COUNT SOURCE KEYS
# ============================================================================

def count_source_keys(path, wanted_name_keys, wanted_address_keys):

    name_counts = Counter()
    address_counts = Counter()

    for chunk in pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        usecols=[
            "entity_id",
            "business_name",
            "business_address",
            "country"
        ],
        chunksize=CHUNK_SIZE
    ):

        for row in chunk.itertuples(index=False):

            for key in name_signatures(
                row.business_name,
                row.country
            ):
                if key in wanted_name_keys:
                    name_counts[key] += 1

            for key in address_signatures(
                row.business_address,
                row.country
            ):
                if key in wanted_address_keys:
                    address_counts[key] += 1

    return name_counts, address_counts


# ============================================================================
# BUILD SOURCE INDEX
# ============================================================================

def build_source_index(
    path,
    allowed_name_keys,
    allowed_address_keys,
    name_counts,
    address_counts
):

    index = defaultdict(list)

    for chunk in pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        usecols=[
            "entity_id",
            "business_name",
            "business_address",
            "country"
        ],
        chunksize=CHUNK_SIZE
    ):

        for row in chunk.itertuples(index=False):

            entity_id = row.entity_id

            for key in name_signatures(
                row.business_name,
                row.country
            ):

                if (
                    key in allowed_name_keys
                    and name_counts[key] <= MAX_KEY_FREQUENCY
                ):
                    index[key].append(entity_id)

            for key in address_signatures(
                row.business_address,
                row.country
            ):

                if (
                    key in allowed_address_keys
                    and address_counts[key] <= MAX_KEY_FREQUENCY
                ):
                    index[key].append(entity_id)

    return index


# ============================================================================
# GENERATE CANDIDATES
# ============================================================================

def generate_candidates(s1, source_index):

    candidates = set()

    for row in s1.itertuples(index=False):

        s1_id = row.entity_id

        keys = []

        keys.extend(
            name_signatures(
                row.business_name,
                row.country
            )
        )

        keys.extend(
            address_signatures(
                row.business_address,
                row.country
            )
        )

        for key in keys:

            for entity_id in source_index.get(key, []):

                candidates.add(
                    (
                        s1_id,
                        entity_id
                    )
                )

    return candidates


# ============================================================================
# MAIN
# ============================================================================

def main():

    start = time.time()

    print("=" * 80)
    print("EXPERIMENT 002B - APPROXIMATE SIGNATURE BLOCKING")
    print("=" * 80)

    print("\nLoading validation S1...")

    s1 = load_validation_s1()

    print(
        f"Validation S1 records: {len(s1):,}"
    )

    print("\nBuilding query signatures...")

    wanted_name_keys, wanted_address_keys = \
        build_query_signatures(s1)

    print(
        f"Unique S1 name signatures: "
        f"{len(wanted_name_keys):,}"
    )

    print(
        f"Unique S1 address signatures: "
        f"{len(wanted_address_keys):,}"
    )

    # ------------------------------------------------------------
    # S2
    # ------------------------------------------------------------

    print("\n" + "-" * 80)
    print("PROCESSING S2")
    print("-" * 80)

    print("\nCounting S2 signatures...")

    s2_name_counts, s2_address_counts = count_source_keys(
        S2_PATH,
        wanted_name_keys,
        wanted_address_keys
    )

    print(
        f"S2 relevant name keys: "
        f"{len(s2_name_counts):,}"
    )

    print(
        f"S2 relevant address keys: "
        f"{len(s2_address_counts):,}"
    )

    print("\nBuilding S2 compact index...")

    s2_index = build_source_index(
        S2_PATH,
        wanted_name_keys,
        wanted_address_keys,
        s2_name_counts,
        s2_address_counts
    )

    print(
        f"S2 indexed signatures: "
        f"{len(s2_index):,}"
    )

    print("\nGenerating S2 candidates...")

    s2_candidates = generate_candidates(
        s1,
        s2_index
    )

    print(
        f"S2 candidate pairs: "
        f"{len(s2_candidates):,}"
    )

    del s2_index

    # ------------------------------------------------------------
    # S3
    # ------------------------------------------------------------

    print("\n" + "-" * 80)
    print("PROCESSING S3")
    print("-" * 80)

    print("\nCounting S3 signatures...")

    s3_name_counts, s3_address_counts = count_source_keys(
        S3_PATH,
        wanted_name_keys,
        wanted_address_keys
    )

    print(
        f"S3 relevant name keys: "
        f"{len(s3_name_counts):,}"
    )

    print(
        f"S3 relevant address keys: "
        f"{len(s3_address_counts):,}"
    )

    print("\nBuilding S3 compact index...")

    s3_index = build_source_index(
        S3_PATH,
        wanted_name_keys,
        wanted_address_keys,
        s3_name_counts,
        s3_address_counts
    )

    print(
        f"S3 indexed signatures: "
        f"{len(s3_index):,}"
    )

    print("\nGenerating S3 candidates...")

    s3_candidates = generate_candidates(
        s1,
        s3_index
    )

    print(
        f"S3 candidate pairs: "
        f"{len(s3_candidates):,}"
    )

    # ------------------------------------------------------------
    # UNION WITH 002A
    # ------------------------------------------------------------

    print("\nLoading existing 002A candidates...")

    existing = pd.read_csv(
        EXISTING_CANDIDATES,
        sep="\t",
        dtype=str,
        keep_default_na=False
    )

    existing_pairs = set(
        zip(
            existing["source1_entity_id"],
            existing["candidate_entity_id"]
        )
    )

    print(
        f"Existing 002A candidates: "
        f"{len(existing_pairs):,}"
    )

    all_candidates = (
        existing_pairs
        | s2_candidates
        | s3_candidates
    )

    print(
        f"\nFinal 002B candidate pairs: "
        f"{len(all_candidates):,}"
    )

    # ------------------------------------------------------------
    # SAVE
    # ------------------------------------------------------------

    output = pd.DataFrame(
        list(all_candidates),
        columns=[
            "source1_entity_id",
            "candidate_entity_id"
        ]
    )

    output.to_csv(
        OUTPUT_PATH,
        sep="\t",
        index=False
    )

    runtime = time.time() - start

    print(
        f"\nOutput: {OUTPUT_PATH}"
    )

    print(
        f"Runtime: {runtime:.2f} seconds"
    )

    print("\n" + "=" * 80)
    print("EXPERIMENT 002B COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
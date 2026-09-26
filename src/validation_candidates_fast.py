from pathlib import Path
from collections import defaultdict, Counter
import re
import unicodedata
import time
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent

TRAIN_DIR = BASE_DIR / "dataset" / "train"
EXP_DIR = BASE_DIR / "experiments" / "001_baseline"
OUT_DIR = BASE_DIR / "experiments" / "009_validation"

S1_PATH = TRAIN_DIR / "train_source1.tsv"
S2_PATH = TRAIN_DIR / "train_source2.tsv"
S3_PATH = TRAIN_DIR / "train_source3.tsv"

VALIDATION_GT = EXP_DIR / "validation_ground_truth.tsv"
OUTPUT = OUT_DIR / "validation_candidate_pairs.tsv"

CHUNK_SIZE = 250_000

# Keep only reasonably selective keys.
MAX_KEY_FREQUENCY = 100

# Minimum token length for name/address token blocking.
MIN_TOKEN_LENGTH = 4


def normalize(value):
    if pd.isna(value):
        return ""

    value = str(value).lower().strip()
    value = unicodedata.normalize("NFKC", value)
    value = re.sub(r"[^\w\s]", " ", value, flags=re.UNICODE)
    value = re.sub(r"\s+", " ", value)

    return value.strip()


def tokens(value):
    text = normalize(value)

    if not text:
        return []

    return [
        x for x in text.split()
        if len(x) >= MIN_TOKEN_LENGTH
    ]


def numbers(value):
    if pd.isna(value):
        return []

    return re.findall(r"\d+", str(value))


def name_keys(name):
    text = normalize(name)

    if not text:
        return set()

    toks = tokens(name)

    keys = set()

    # Individual informative name tokens
    for token in toks:
        keys.add(("NAME_TOKEN", token))

    # Two-token combinations
    informative = [
        x for x in toks
        if len(x) >= 5
    ]

    for i in range(min(len(informative), 6)):
        for j in range(i + 1, min(len(informative), 6)):
            keys.add(
                (
                    "NAME_PAIR",
                    informative[i],
                    informative[j],
                )
            )

    return keys


def address_keys(address):
    text = normalize(address)

    if not text:
        return set()

    toks = tokens(address)
    nums = numbers(address)

    keys = set()

    # Number + informative token
    if nums:
        house = nums[0]

        for token in toks:
            if len(token) >= 5:
                keys.add(
                    (
                        "ADDR_NUM_TOKEN",
                        house,
                        token,
                    )
                )

    # Two informative address tokens
    informative = [
        x for x in toks
        if len(x) >= 6
    ]

    for i in range(min(len(informative), 6)):
        for j in range(i + 1, min(len(informative), 6)):
            keys.add(
                (
                    "ADDR_PAIR",
                    informative[i],
                    informative[j],
                )
            )

    return keys


def load_validation_s1():

    gt = pd.read_csv(
        VALIDATION_GT,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        usecols=["source1_entity_id"],
    )

    validation_ids = set(
        gt["source1_entity_id"]
    )

    s1 = pd.read_csv(
        S1_PATH,
        sep="\t",
        dtype=str,
        keep_default_na=False,
    )

    s1 = s1[
        s1["entity_id"].isin(validation_ids)
    ].copy()

    return s1


def build_query_keys(s1):

    name_keys_set = set()
    address_keys_set = set()

    print("\nBuilding validation S1 signatures...")

    for row in s1.itertuples(index=False):

        name_keys_set.update(
            name_keys(row.business_name)
        )

        address_keys_set.update(
            address_keys(row.business_address)
        )

    print(
        f"Unique name keys: {len(name_keys_set):,}"
    )

    print(
        f"Unique address keys: {len(address_keys_set):,}"
    )

    return name_keys_set, address_keys_set


def build_source_index(
    path,
    wanted_name_keys,
    wanted_address_keys,
):

    print(f"\nScanning {path.name}...")

    counts = Counter()

    # ------------------------------------------------------------
    # PASS 1 — frequency counting
    # ------------------------------------------------------------

    rows = 0

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
        chunksize=CHUNK_SIZE,
    ):

        rows += len(chunk)

        for row in chunk.itertuples(index=False):

            keys = (
                name_keys(row.business_name)
                |
                address_keys(row.business_address)
            )

            for key in keys:

                if (
                    key in wanted_name_keys
                    or key in wanted_address_keys
                ):
                    counts[key] += 1

    print(
        f"Rows scanned: {rows:,}"
    )

    print(
        f"Relevant keys: {len(counts):,}"
    )

    # ------------------------------------------------------------
    # PASS 2 — build selective index
    # ------------------------------------------------------------

    index = defaultdict(list)

    rows = 0

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
        chunksize=CHUNK_SIZE,
    ):

        rows += len(chunk)

        for row in chunk.itertuples(index=False):

            entity_id = row.entity_id

            keys = (
                name_keys(row.business_name)
                |
                address_keys(row.business_address)
            )

            for key in keys:

                if key not in counts:
                    continue

                if counts[key] <= MAX_KEY_FREQUENCY:
                    index[key].append(entity_id)

    print(
        f"Indexed keys: {len(index):,}"
    )

    return index


def generate_candidates(
    s1,
    source_index,
):

    pairs = set()

    for row in s1.itertuples(index=False):

        s1_id = row.entity_id

        keys = (
            name_keys(row.business_name)
            |
            address_keys(row.business_address)
        )

        for key in keys:

            for candidate_id in source_index.get(
                key,
                []
            ):

                pairs.add(
                    (
                        s1_id,
                        candidate_id,
                    )
                )

    return pairs


def main():

    start = time.time()

    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 80)
    print("STAGE 9A - FAST VALIDATION CANDIDATE GENERATION")
    print("=" * 80)

    # ------------------------------------------------------------
    # Validation S1
    # ------------------------------------------------------------

    s1 = load_validation_s1()

    print(
        f"\nValidation S1 records: {len(s1):,}"
    )

    # ------------------------------------------------------------
    # Query signatures
    # ------------------------------------------------------------

    wanted_name_keys, wanted_address_keys = (
        build_query_keys(s1)
    )

    # ------------------------------------------------------------
    # S2
    # ------------------------------------------------------------

    print("\n" + "-" * 80)
    print("PROCESSING S2")
    print("-" * 80)

    s2_index = build_source_index(
        S2_PATH,
        wanted_name_keys,
        wanted_address_keys,
    )

    print("\nGenerating S2 candidates...")

    s2_pairs = generate_candidates(
        s1,
        s2_index,
    )

    print(
        f"S2 candidate pairs: {len(s2_pairs):,}"
    )

    del s2_index

    # ------------------------------------------------------------
    # S3
    # ------------------------------------------------------------

    print("\n" + "-" * 80)
    print("PROCESSING S3")
    print("-" * 80)

    s3_index = build_source_index(
        S3_PATH,
        wanted_name_keys,
        wanted_address_keys,
    )

    print("\nGenerating S3 candidates...")

    s3_pairs = generate_candidates(
        s1,
        s3_index,
    )

    print(
        f"S3 candidate pairs: {len(s3_pairs):,}"
    )

    del s3_index

    # ------------------------------------------------------------
    # Combine
    # ------------------------------------------------------------

    all_pairs = s2_pairs | s3_pairs

    print("\n" + "=" * 80)
    print(
        f"FINAL VALIDATION CANDIDATE PAIRS: "
        f"{len(all_pairs):,}"
    )
    print("=" * 80)

    output = pd.DataFrame(
        list(all_pairs),
        columns=[
            "source1_entity_id",
            "candidate_entity_id",
        ],
    )

    output.to_csv(
        OUTPUT,
        sep="\t",
        index=False,
    )

    print(
        f"\nOutput: {OUTPUT}"
    )

    print(
        f"Runtime: {time.time() - start:.2f} seconds"
    )

    print("\nSTATUS: PASS")


if __name__ == "__main__":
    main()
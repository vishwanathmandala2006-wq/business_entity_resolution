from pathlib import Path
from collections import Counter, defaultdict
import re
import unicodedata
import pandas as pd


# ============================================================================
# CONFIG
# ============================================================================

BASE_DIR = Path(__file__).resolve().parent.parent

TRAIN_DIR = BASE_DIR / "dataset" / "train"
EXP_DIR = BASE_DIR / "experiments" / "001_baseline"

S1_PATH = TRAIN_DIR / "train_source1.tsv"
S2_PATH = TRAIN_DIR / "train_source2.tsv"
S3_PATH = TRAIN_DIR / "train_source3.tsv"

MISSED_PATH = EXP_DIR / "missed_002b_pairs.tsv"
ROUTE_RESULT_PATH = EXP_DIR / "route_recovery_002c.tsv"

SAMPLE_SIZE = 10_000
CHUNK_SIZE = 250_000

# Frequency thresholds we want to test.
THRESHOLDS = [
    10,
    25,
    50,
    100,
    250,
    500,
    1_000,
    5_000,
    10_000,
]


# ============================================================================
# NORMALIZATION
# ============================================================================

LEGAL_SUFFIXES = {
    "inc",
    "incorporated",
    "corp",
    "corporation",
    "co",
    "company",
    "llc",
    "llp",
    "ltd",
    "limited",
    "plc",
    "pc",
    "pvt",
    "private",
    "public",
    "lp",
}


def normalize_unicode(value):
    if pd.isna(value):
        return ""

    value = str(value).lower().strip()
    value = unicodedata.normalize("NFKC", value)

    value = re.sub(
        r"[^\w\s]",
        " ",
        value,
        flags=re.UNICODE,
    )

    value = re.sub(r"\s+", " ", value)

    return value.strip()


def tokens(value):
    return normalize_unicode(value).split()


def name_tokens(value):
    """
    Informative name tokens.

    Legal suffixes are removed because they are poor blocking signals.
    """
    return {
        token
        for token in tokens(value)
        if len(token) >= 3
        and token not in LEGAL_SUFFIXES
    }


def address_tokens(value):
    """
    Informative address tokens.

    Numeric tokens are excluded here because number overlap
    is measured separately.
    """
    return {
        token
        for token in tokens(value)
        if len(token) >= 4
        and not token.isdigit()
    }


# ============================================================================
# LOAD MISSED SAMPLE
# ============================================================================

def load_missed_sample():

    missed = pd.read_csv(
        MISSED_PATH,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        nrows=SAMPLE_SIZE,
    )

    print(f"Missed sample: {len(missed):,}")

    return missed


# ============================================================================
# LOAD REQUIRED RECORDS
# ============================================================================

def load_required_records(missed):

    s1_ids = set(missed["source1_entity_id"])
    candidate_ids = set(missed["candidate_entity_id"])

    print(f"S1 IDs required:        {len(s1_ids):,}")
    print(f"Candidate IDs required: {len(candidate_ids):,}")

    # ------------------------------------------------------------------------
    # S1
    # ------------------------------------------------------------------------

    s1 = pd.read_csv(
        S1_PATH,
        sep="\t",
        dtype=str,
        keep_default_na=False,
    )

    s1 = s1[
        s1["entity_id"].isin(s1_ids)
    ].copy()

    # ------------------------------------------------------------------------
    # S2 + S3
    # ------------------------------------------------------------------------

    remaining = set(candidate_ids)
    candidate_frames = []

    for path, source in [
        (S2_PATH, "S2"),
        (S3_PATH, "S3"),
    ]:

        print(f"\nScanning {source}...")

        for chunk in pd.read_csv(
            path,
            sep="\t",
            dtype=str,
            keep_default_na=False,
            chunksize=CHUNK_SIZE,
        ):

            found = chunk[
                chunk["entity_id"].isin(remaining)
            ].copy()

            if len(found):

                found["source"] = source

                candidate_frames.append(found)

                remaining -= set(
                    found["entity_id"]
                )

            if not remaining:
                break

    candidates = pd.concat(
        candidate_frames,
        ignore_index=True,
    )

    print(
        f"\nCandidate records loaded: "
        f"{len(candidates):,}"
    )

    if remaining:
        print(
            f"WARNING: {len(remaining):,} candidate IDs "
            f"were not found."
        )

    return s1, candidates


# ============================================================================
# BUILD GLOBAL TOKEN FREQUENCIES
# ============================================================================

def count_token_frequencies():

    name_counter = Counter()
    address_counter = Counter()

    print("\n" + "=" * 80)
    print("COUNTING GLOBAL TOKEN FREQUENCIES")
    print("=" * 80)

    for path, source in [
        (S1_PATH, "S1"),
        (S2_PATH, "S2"),
        (S3_PATH, "S3"),
    ]:

        print(f"\nProcessing {source}...")

        for chunk in pd.read_csv(
            path,
            sep="\t",
            dtype=str,
            keep_default_na=False,
            chunksize=CHUNK_SIZE,
        ):

            for value in chunk["business_name"]:
                name_counter.update(
                    name_tokens(value)
                )

            for value in chunk["business_address"]:
                address_counter.update(
                    address_tokens(value)
                )

    return name_counter, address_counter


# ============================================================================
# GET SHARED TOKENS FOR EACH MISSED PAIR
# ============================================================================

def build_pair_token_data(
    missed,
    s1,
    candidates,
):

    s1_map = {
        row.entity_id: row
        for row in s1.itertuples(index=False)
    }

    candidate_map = {
        row.entity_id: row
        for row in candidates.itertuples(index=False)
    }

    pair_data = []

    missing_records = 0

    for pair in missed.itertuples(index=False):

        s1_row = s1_map.get(
            pair.source1_entity_id
        )

        candidate_row = candidate_map.get(
            pair.candidate_entity_id
        )

        if (
            s1_row is None
            or candidate_row is None
        ):
            missing_records += 1
            continue

        n1 = name_tokens(
            s1_row.business_name
        )

        n2 = name_tokens(
            candidate_row.business_name
        )

        a1 = address_tokens(
            s1_row.business_address
        )

        a2 = address_tokens(
            candidate_row.business_address
        )

        shared_name = n1 & n2
        shared_address = a1 & a2

        pair_data.append({
            "source1_entity_id":
                pair.source1_entity_id,

            "candidate_entity_id":
                pair.candidate_entity_id,

            "shared_name_tokens":
                tuple(sorted(shared_name)),

            "shared_address_tokens":
                tuple(sorted(shared_address)),
        })

    if missing_records:
        print(
            f"WARNING: {missing_records:,} pairs "
            f"could not be resolved."
        )

    return pair_data


# ============================================================================
# FREQUENCY DISTRIBUTION
# ============================================================================

def frequency_stats(
    token_sets,
    counter,
):

    rows = []

    for tokens_set in token_sets:

        frequencies = [
            counter[token]
            for token in tokens_set
            if token in counter
        ]

        if not frequencies:
            rows.append({
                "min_freq": None,
                "max_freq": None,
                "best_freq": None,
            })
            continue

        rows.append({
            "min_freq": min(frequencies),
            "max_freq": max(frequencies),
            "best_freq": min(frequencies),
        })

    return rows


# ============================================================================
# RECOVERY UNDER FREQUENCY CAP
# ============================================================================

def calculate_recovery(
    pair_data,
    counter,
    token_field,
    thresholds,
):

    results = []

    total = len(pair_data)

    for threshold in thresholds:

        recovered = 0

        for pair in pair_data:

            shared = pair[token_field]

            if any(
                counter[token] <= threshold
                for token in shared
            ):
                recovered += 1

        recovery = (
            recovered / total * 100
            if total
            else 0
        )

        results.append({
            "threshold": threshold,
            "recovered": recovered,
            "recovery_pct": recovery,
        })

    return pd.DataFrame(results)


# ============================================================================
# TOKEN-LEVEL ANALYSIS
# ============================================================================

def most_useful_tokens(
    pair_data,
    counter,
    token_field,
    top_n=50,
):

    occurrence = Counter()

    for pair in pair_data:

        for token in pair[token_field]:

            occurrence[token] += 1

    rows = []

    for token, true_pair_count in occurrence.most_common(top_n):

        rows.append({
            "token": token,
            "true_pair_occurrences":
                true_pair_count,
            "global_frequency":
                counter[token],
        })

    return pd.DataFrame(rows)


# ============================================================================
# MAIN
# ============================================================================

def main():

    print("=" * 80)
    print("002C TOKEN FREQUENCY ANALYSIS")
    print("=" * 80)

    # ------------------------------------------------------------------------
    # 1. Load sample
    # ------------------------------------------------------------------------

    missed = load_missed_sample()

    # ------------------------------------------------------------------------
    # 2. Load records
    # ------------------------------------------------------------------------

    s1, candidates = load_required_records(
        missed
    )

    # ------------------------------------------------------------------------
    # 3. Global frequencies
    # ------------------------------------------------------------------------

    name_counter, address_counter = (
        count_token_frequencies()
    )

    print(
        f"\nUnique name tokens: "
        f"{len(name_counter):,}"
    )

    print(
        f"Unique address tokens: "
        f"{len(address_counter):,}"
    )

    # ------------------------------------------------------------------------
    # 4. Shared tokens per true pair
    # ------------------------------------------------------------------------

    pair_data = build_pair_token_data(
        missed,
        s1,
        candidates,
    )

    print(
        f"\nPairs successfully analyzed: "
        f"{len(pair_data):,}"
    )

    # ------------------------------------------------------------------------
    # 5. Recovery curves
    # ------------------------------------------------------------------------

    name_recovery = calculate_recovery(
        pair_data,
        name_counter,
        "shared_name_tokens",
        THRESHOLDS,
    )

    address_recovery = calculate_recovery(
        pair_data,
        address_counter,
        "shared_address_tokens",
        THRESHOLDS,
    )

    print("\n" + "=" * 80)
    print("NAME TOKEN FREQUENCY CAP")
    print("=" * 80)

    print(
        name_recovery.to_string(
            index=False,
            formatters={
                "recovery_pct":
                    lambda x: f"{x:.2f}%"
            },
        )
    )

    print("\n" + "=" * 80)
    print("ADDRESS TOKEN FREQUENCY CAP")
    print("=" * 80)

    print(
        address_recovery.to_string(
            index=False,
            formatters={
                "recovery_pct":
                    lambda x: f"{x:.2f}%"
            },
        )
    )

    # ------------------------------------------------------------------------
    # 6. Combined name + address recovery
    # ------------------------------------------------------------------------

    print("\n" + "=" * 80)
    print("COMBINED TOKEN RECOVERY")
    print("=" * 80)

    combined_rows = []

    total = len(pair_data)

    for threshold in THRESHOLDS:

        recovered_name = 0
        recovered_address = 0
        recovered_any = 0
        recovered_both = 0

        for pair in pair_data:

            name_ok = any(
                name_counter[token] <= threshold
                for token in pair[
                    "shared_name_tokens"
                ]
            )

            address_ok = any(
                address_counter[token] <= threshold
                for token in pair[
                    "shared_address_tokens"
                ]
            )

            if name_ok:
                recovered_name += 1

            if address_ok:
                recovered_address += 1

            if name_ok or address_ok:
                recovered_any += 1

            if name_ok and address_ok:
                recovered_both += 1

        combined_rows.append({
            "threshold": threshold,

            "name_recovered":
                recovered_name,

            "name_pct":
                recovered_name / total * 100,

            "address_recovered":
                recovered_address,

            "address_pct":
                recovered_address / total * 100,

            "any_recovered":
                recovered_any,

            "any_pct":
                recovered_any / total * 100,

            "both_recovered":
                recovered_both,

            "both_pct":
                recovered_both / total * 100,
        })

    combined_df = pd.DataFrame(
        combined_rows
    )

    print(
        combined_df.to_string(
            index=False,
            formatters={
                "name_pct":
                    lambda x: f"{x:.2f}%",
                "address_pct":
                    lambda x: f"{x:.2f}%",
                "any_pct":
                    lambda x: f"{x:.2f}%",
                "both_pct":
                    lambda x: f"{x:.2f}%",
            },
        )
    )

    # ------------------------------------------------------------------------
    # 7. Most useful name tokens
    # ------------------------------------------------------------------------

    name_useful = most_useful_tokens(
        pair_data,
        name_counter,
        "shared_name_tokens",
        top_n=50,
    )

    address_useful = most_useful_tokens(
        pair_data,
        address_counter,
        "shared_address_tokens",
        top_n=50,
    )

    # ------------------------------------------------------------------------
    # 8. Save outputs
    # ------------------------------------------------------------------------

    name_recovery.to_csv(
        EXP_DIR / "002c_name_frequency_recovery.tsv",
        sep="\t",
        index=False,
    )

    address_recovery.to_csv(
        EXP_DIR / "002c_address_frequency_recovery.tsv",
        sep="\t",
        index=False,
    )

    combined_df.to_csv(
        EXP_DIR / "002c_combined_frequency_recovery.tsv",
        sep="\t",
        index=False,
    )

    name_useful.to_csv(
        EXP_DIR / "002c_useful_name_tokens.tsv",
        sep="\t",
        index=False,
    )

    address_useful.to_csv(
        EXP_DIR / "002c_useful_address_tokens.tsv",
        sep="\t",
        index=False,
    )

    print("\n" + "=" * 80)
    print("TOP SHARED NAME TOKENS")
    print("=" * 80)

    print(
        name_useful.to_string(
            index=False
        )
    )

    print("\n" + "=" * 80)
    print("TOP SHARED ADDRESS TOKENS")
    print("=" * 80)

    print(
        address_useful.to_string(
            index=False
        )
    )

    print("\n" + "=" * 80)
    print("FILES WRITTEN")
    print("=" * 80)

    print(
        EXP_DIR /
        "002c_name_frequency_recovery.tsv"
    )

    print(
        EXP_DIR /
        "002c_address_frequency_recovery.tsv"
    )

    print(
        EXP_DIR /
        "002c_combined_frequency_recovery.tsv"
    )

    print(
        EXP_DIR /
        "002c_useful_name_tokens.tsv"
    )

    print(
        EXP_DIR /
        "002c_useful_address_tokens.tsv"
    )


if __name__ == "__main__":
    main()
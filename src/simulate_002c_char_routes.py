from pathlib import Path
from collections import Counter
import re
import unicodedata
import numpy as np
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

SAMPLE_SIZE = 10_000
CHUNK_SIZE = 250_000

# A character signature is usable only if its global
# frequency is <= one of these caps.
FREQ_CAPS = [
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

def normalize_unicode(value):

    if pd.isna(value):
        return ""

    value = str(value).lower().strip()

    value = unicodedata.normalize(
        "NFKC",
        value,
    )

    value = re.sub(
        r"\s+",
        " ",
        value,
    )

    return value.strip()


def compact(value):

    value = normalize_unicode(value)

    return re.sub(
        r"[^\w]",
        "",
        value,
        flags=re.UNICODE,
    )


def alnum_space(value):

    value = normalize_unicode(value)

    return re.sub(
        r"[^\w\s]",
        " ",
        value,
        flags=re.UNICODE,
    )


# ============================================================================
# CHARACTER SIGNATURES
# ============================================================================

def char_ngrams(value, n):

    value = alnum_space(value)

    if len(value) < n:
        return set()

    return {
        value[i:i+n]
        for i in range(len(value) - n + 1)
    }


def compact_char_ngrams(value, n):

    value = compact(value)

    if len(value) < n:
        return set()

    return {
        value[i:i+n]
        for i in range(len(value) - n + 1)
    }


def sorted_char_ngrams(value, n):

    grams = compact_char_ngrams(
        value,
        n,
    )

    return {
        "".join(sorted(g))
        for g in grams
    }


def digit_aware_ngrams(value, n=3):

    value = normalize_unicode(value)

    # Keep digits, letters and spaces.
    value = re.sub(
        r"[^\w\s]",
        "",
        value,
        flags=re.UNICODE,
    )

    # Replace runs of digits with a marker while
    # preserving the fact that a numeric component exists.
    value = re.sub(
        r"\d+",
        "#",
        value,
    )

    value = re.sub(
        r"\s+",
        " ",
        value,
    ).strip()

    if len(value) < n:
        return set()

    return {
        value[i:i+n]
        for i in range(len(value) - n + 1)
    }


# ============================================================================
# LOAD MISSED SAMPLE
# ============================================================================

def load_missed():

    missed = pd.read_csv(
        MISSED_PATH,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        nrows=SAMPLE_SIZE,
    )

    print(
        f"Missed pairs loaded: {len(missed):,}"
    )

    return missed


# ============================================================================
# LOAD RECORDS
# ============================================================================

def load_records(missed):

    s1_ids = set(
        missed["source1_entity_id"]
    )

    candidate_ids = set(
        missed["candidate_entity_id"]
    )

    print(
        f"S1 IDs:        {len(s1_ids):,}"
    )

    print(
        f"Candidate IDs: {len(candidate_ids):,}"
    )

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
    # S2/S3
    # ------------------------------------------------------------------------

    remaining = set(candidate_ids)

    frames = []

    for path, source in [
        (S2_PATH, "S2"),
        (S3_PATH, "S3"),
    ]:

        print(
            f"\nScanning {source}..."
        )

        for chunk in pd.read_csv(
            path,
            sep="\t",
            dtype=str,
            keep_default_na=False,
            chunksize=CHUNK_SIZE,
        ):

            found = chunk[
                chunk["entity_id"].isin(
                    remaining
                )
            ].copy()

            if len(found):

                found["source"] = source

                frames.append(found)

                remaining -= set(
                    found["entity_id"]
                )

            if not remaining:
                break

    candidates = pd.concat(
        frames,
        ignore_index=True,
    )

    print(
        f"\nCandidates loaded: "
        f"{len(candidates):,}"
    )

    return s1, candidates


# ============================================================================
# BUILD GLOBAL CHARACTER-SIGNATURE FREQUENCIES
# ============================================================================

ROUTES = [
    "NAME_CHAR2",
    "NAME_CHAR3",
    "NAME_CHAR4",
    "NAME_COMPACT_CHAR3",
    "NAME_COMPACT_CHAR4",
    "NAME_SORTED_CHAR3",
    "ADDRESS_CHAR3",
    "ADDRESS_CHAR4",
    "ADDRESS_COMPACT_CHAR3",
    "ADDRESS_COMPACT_CHAR4",
    "ADDRESS_SORTED_CHAR3",
    "ADDRESS_DIGIT_CHAR3",
]


def get_route_signatures(row, route):

    name = row.business_name
    address = row.business_address

    if route == "NAME_CHAR2":
        return char_ngrams(name, 2)

    if route == "NAME_CHAR3":
        return char_ngrams(name, 3)

    if route == "NAME_CHAR4":
        return char_ngrams(name, 4)

    if route == "NAME_COMPACT_CHAR3":
        return compact_char_ngrams(name, 3)

    if route == "NAME_COMPACT_CHAR4":
        return compact_char_ngrams(name, 4)

    if route == "NAME_SORTED_CHAR3":
        return sorted_char_ngrams(name, 3)

    if route == "ADDRESS_CHAR3":
        return char_ngrams(address, 3)

    if route == "ADDRESS_CHAR4":
        return char_ngrams(address, 4)

    if route == "ADDRESS_COMPACT_CHAR3":
        return compact_char_ngrams(address, 3)

    if route == "ADDRESS_COMPACT_CHAR4":
        return compact_char_ngrams(address, 4)

    if route == "ADDRESS_SORTED_CHAR3":
        return sorted_char_ngrams(address, 3)

    if route == "ADDRESS_DIGIT_CHAR3":
        return digit_aware_ngrams(address, 3)

    raise ValueError(
        f"Unknown route: {route}"
    )


def count_global_signature_frequencies():

    counters = {
        route: Counter()
        for route in ROUTES
    }

    print("\n" + "=" * 80)
    print("COUNTING GLOBAL CHARACTER SIGNATURE FREQUENCIES")
    print("=" * 80)

    for path, source in [
        (S1_PATH, "S1"),
        (S2_PATH, "S2"),
        (S3_PATH, "S3"),
    ]:

        print(
            f"\nProcessing {source}..."
        )

        for chunk in pd.read_csv(
            path,
            sep="\t",
            dtype=str,
            keep_default_na=False,
            chunksize=CHUNK_SIZE,
        ):

            for row in chunk.itertuples(
                index=False
            ):

                for route in ROUTES:

                    signatures = (
                        get_route_signatures(
                            row,
                            route,
                        )
                    )

                    counters[route].update(
                        signatures
                    )

    for route in ROUTES:

        print(
            f"{route:<30} "
            f"{len(counters[route]):,} unique signatures"
        )

    return counters


# ============================================================================
# TRUE-PAIR SIGNATURE OVERLAP
# ============================================================================

def build_pair_data(
    missed,
    s1,
    candidates,
):

    s1_map = {
        row.entity_id: row
        for row in s1.itertuples(
            index=False
        )
    }

    candidate_map = {
        row.entity_id: row
        for row in candidates.itertuples(
            index=False
        )
    }

    rows = []

    for pair in missed.itertuples(
        index=False
    ):

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
            continue

        row = {
            "source1_entity_id":
                pair.source1_entity_id,

            "candidate_entity_id":
                pair.candidate_entity_id,

            "source":
                getattr(
                    candidate_row,
                    "source",
                    "",
                ),
        }

        for route in ROUTES:

            a = get_route_signatures(
                s1_row,
                route,
            )

            b = get_route_signatures(
                candidate_row,
                route,
            )

            row[route] = a & b

        rows.append(row)

    return rows


# ============================================================================
# RECOVERY
# ============================================================================

def evaluate_recovery(
    pair_data,
    counters,
    route,
    cap,
):

    recovered = 0

    for pair in pair_data:

        shared = pair[route]

        valid = [
            sig
            for sig in shared
            if counters[route][sig] <= cap
        ]

        if valid:
            recovered += 1

    return recovered


# ============================================================================
# SAMPLE INDEX
# ============================================================================

def build_sample_index(
    candidates,
    route,
    counter,
    cap,
):

    index = {}

    for row in candidates.itertuples(
        index=False
    ):

        signatures = get_route_signatures(
            row,
            route,
        )

        for signature in signatures:

            if counter[signature] > cap:
                continue

            index.setdefault(
                signature,
                set(),
            ).add(
                row.entity_id
            )

    return index


# ============================================================================
# SAMPLE CANDIDATE MULTIPLICITY
# ============================================================================

def estimate_multiplicity(
    pair_data,
    s1,
    candidates,
    route,
    counter,
    cap,
):

    index = build_sample_index(
        candidates,
        route,
        counter,
        cap,
    )

    s1_map = {
        row.entity_id: row
        for row in s1.itertuples(
            index=False
        )
    }

    counts = []
    covered = 0

    for pair in pair_data:

        s1_row = s1_map[
            pair["source1_entity_id"]
        ]

        signatures = get_route_signatures(
            s1_row,
            route,
        )

        candidate_ids = set()

        for signature in signatures:

            if counter[signature] > cap:
                continue

            candidate_ids.update(
                index.get(
                    signature,
                    set(),
                )
            )

        count = len(candidate_ids)

        counts.append(count)

        if pair[
            "candidate_entity_id"
        ] in candidate_ids:

            covered += 1

    if not counts:

        return {
            "median": 0,
            "p90": 0,
            "p95": 0,
            "p99": 0,
            "max": 0,
            "mean": 0,
            "covered": 0,
            "coverage_pct": 0,
        }

    arr = np.array(
        counts,
        dtype=np.int64,
    )

    return {
        "median":
            float(np.median(arr)),

        "p90":
            float(np.percentile(arr, 90)),

        "p95":
            float(np.percentile(arr, 95)),

        "p99":
            float(np.percentile(arr, 99)),

        "max":
            int(arr.max()),

        "mean":
            float(arr.mean()),

        "covered":
            covered,

        "coverage_pct":
            covered /
            len(pair_data) *
            100,
    }


# ============================================================================
# MAIN
# ============================================================================

def main():

    print("=" * 80)
    print("002C CHARACTER SIGNATURE SIMULATION")
    print("=" * 80)

    # ------------------------------------------------------------------------
    # Load diagnostic sample
    # ------------------------------------------------------------------------

    missed = load_missed()

    s1, candidates = load_records(
        missed
    )

    # ------------------------------------------------------------------------
    # Global signature frequencies
    # ------------------------------------------------------------------------

    counters = (
        count_global_signature_frequencies()
    )

    # ------------------------------------------------------------------------
    # Pair overlap
    # ------------------------------------------------------------------------

    pair_data = build_pair_data(
        missed,
        s1,
        candidates,
    )

    print(
        f"\nTrue pairs analyzed: "
        f"{len(pair_data):,}"
    )

    # ------------------------------------------------------------------------
    # Evaluate routes
    # ------------------------------------------------------------------------

    results = []

    for route in ROUTES:

        print("\n" + "=" * 80)
        print(route)
        print("=" * 80)

        for cap in FREQ_CAPS:

            recovered = evaluate_recovery(
                pair_data,
                counters,
                route,
                cap,
            )

            recovery_pct = (
                recovered /
                len(pair_data) *
                100
            )

            stats = estimate_multiplicity(
                pair_data,
                s1,
                candidates,
                route,
                counters[route],
                cap,
            )

            result = {
                "route": route,
                "cap": cap,

                "recovered":
                    recovered,

                "recovery_pct":
                    recovery_pct,

                "median_candidates":
                    stats["median"],

                "p90_candidates":
                    stats["p90"],

                "p95_candidates":
                    stats["p95"],

                "p99_candidates":
                    stats["p99"],

                "max_candidates":
                    stats["max"],

                "mean_candidates":
                    stats["mean"],

                "sample_coverage":
                    stats["coverage_pct"],
            }

            results.append(result)

            print(
                f"cap={cap:>6,} "
                f"recovery={recovery_pct:6.2f}% "
                f"median={stats['median']:7.1f} "
                f"p95={stats['p95']:7.1f} "
                f"p99={stats['p99']:7.1f} "
                f"max={stats['max']:7d}"
            )

    # ------------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------------

    result_df = pd.DataFrame(
        results
    )

    output_path = (
        EXP_DIR /
        "002c_character_route_results.tsv"
    )

    result_df.to_csv(
        output_path,
        sep="\t",
        index=False,
    )

    print("\n" + "=" * 80)
    print("BEST RECOVERY BY CHARACTER ROUTE")
    print("=" * 80)

    for route in ROUTES:

        subset = result_df[
            result_df["route"] == route
        ]

        best = subset.loc[
            subset["recovery_pct"].idxmax()
        ]

        print(
            f"{route:<30} "
            f"cap={int(best['cap']):>6,} "
            f"recovery={best['recovery_pct']:6.2f}% "
            f"median={best['median_candidates']:7.1f} "
            f"p95={best['p95_candidates']:7.1f} "
            f"p99={best['p99_candidates']:7.1f}"
        )

    print(
        f"\nResult file:\n{output_path}"
    )

    print(
        "\nCharacter signature simulation complete."
    )


if __name__ == "__main__":
    main()
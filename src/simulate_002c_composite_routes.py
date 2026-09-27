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

# Maximum frequency of an individual component.
# We test several caps to understand selectivity.
FREQ_CAPS = [
    25,
    50,
    100,
    250,
    500,
    1_000,
    5_000,
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

    value = unicodedata.normalize(
        "NFKC",
        value,
    )

    value = re.sub(
        r"[^\w\s]",
        " ",
        value,
        flags=re.UNICODE,
    )

    value = re.sub(
        r"\s+",
        " ",
        value,
    )

    return value.strip()


def tokens(value):

    value = normalize_unicode(value)

    if not value:
        return []

    return value.split()


def name_tokens(value):

    return {
        token
        for token in tokens(value)
        if len(token) >= 3
        and token not in LEGAL_SUFFIXES
    }


def address_tokens(value):

    return {
        token
        for token in tokens(value)
        if len(token) >= 4
        and not token.isdigit()
    }


def address_numbers(value):

    if pd.isna(value):
        return set()

    return set(
        re.findall(
            r"\d+",
            str(value),
        )
    )


def name_core_tokens(value):

    return name_tokens(value)


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
# LOAD REQUIRED RECORDS
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
# BUILD RECORD COMPONENTS
# ============================================================================

def build_components(df):

    records = {}

    for row in df.itertuples(
        index=False
    ):

        records[row.entity_id] = {
            "source": getattr(
                row,
                "source",
                "S1",
            ),

            "country": normalize_unicode(
                row.country
            ),

            "name_tokens":
                name_tokens(
                    row.business_name
                ),

            "address_tokens":
                address_tokens(
                    row.business_address
                ),

            "numbers":
                address_numbers(
                    row.business_address
                ),
        }

    return records


# ============================================================================
# GLOBAL FREQUENCIES
# ============================================================================

def count_global_components():

    name_freq = Counter()
    address_freq = Counter()
    number_freq = Counter()

    print("\n" + "=" * 80)
    print("BUILDING GLOBAL COMPONENT FREQUENCIES")
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

            for value in chunk[
                "business_name"
            ]:

                name_freq.update(
                    name_tokens(value)
                )

            for value in chunk[
                "business_address"
            ]:

                address_freq.update(
                    address_tokens(value)
                )

                number_freq.update(
                    address_numbers(value)
                )

    print(
        f"\nUnique name components: "
        f"{len(name_freq):,}"
    )

    print(
        f"Unique address components: "
        f"{len(address_freq):,}"
    )

    print(
        f"Unique number components: "
        f"{len(number_freq):,}"
    )

    return (
        name_freq,
        address_freq,
        number_freq,
    )


# ============================================================================
# TRUE PAIR COMPONENT OVERLAP
# ============================================================================

def build_pair_data(
    missed,
    s1_records,
    candidate_records,
):

    rows = []

    for pair in missed.itertuples(
        index=False
    ):

        s1 = s1_records.get(
            pair.source1_entity_id
        )

        candidate = candidate_records.get(
            pair.candidate_entity_id
        )

        if s1 is None or candidate is None:
            continue

        shared_name = (
            s1["name_tokens"]
            &
            candidate["name_tokens"]
        )

        shared_address = (
            s1["address_tokens"]
            &
            candidate["address_tokens"]
        )

        shared_numbers = (
            s1["numbers"]
            &
            candidate["numbers"]
        )

        rows.append({
            "source1_entity_id":
                pair.source1_entity_id,

            "candidate_entity_id":
                pair.candidate_entity_id,

            "source":
                candidate["source"],

            "country":
                s1["country"],

            "shared_name":
                tuple(
                    sorted(shared_name)
                ),

            "shared_address":
                tuple(
                    sorted(shared_address)
                ),

            "shared_numbers":
                tuple(
                    sorted(shared_numbers)
                ),
        })

    return rows


# ============================================================================
# TRUE-PAIR RECOVERY
# ============================================================================

def evaluate_route(
    pair_data,
    route,
    name_freq,
    address_freq,
    number_freq,
    cap,
):

    recovered = 0

    for pair in pair_data:

        name = pair["shared_name"]
        address = pair["shared_address"]
        numbers = pair["shared_numbers"]

        if route == "NAME_NAME":

            valid = [
                token
                for token in name
                if name_freq[token] <= cap
            ]

            # Need at least TWO independently
            # shared name components.
            recovered_here = (
                len(valid) >= 2
            )

        elif route == "ADDRESS_ADDRESS":

            valid = [
                token
                for token in address
                if address_freq[token] <= cap
            ]

            recovered_here = (
                len(valid) >= 2
            )

        elif route == "ADDRESS_NUMBER":

            valid_address = [
                token
                for token in address
                if address_freq[token] <= cap
            ]

            valid_numbers = [
                number
                for number in numbers
                if number_freq[number] <= cap
            ]

            recovered_here = (
                len(valid_address) >= 1
                and
                len(valid_numbers) >= 1
            )

        elif route == "NAME_ADDRESS":

            valid_name = [
                token
                for token in name
                if name_freq[token] <= cap
            ]

            valid_address = [
                token
                for token in address
                if address_freq[token] <= cap
            ]

            recovered_here = (
                len(valid_name) >= 1
                and
                len(valid_address) >= 1
            )

        elif route == "NAME_NUMBER":

            valid_name = [
                token
                for token in name
                if name_freq[token] <= cap
            ]

            valid_numbers = [
                number
                for number in numbers
                if number_freq[number] <= cap
            ]

            recovered_here = (
                len(valid_name) >= 1
                and
                len(valid_numbers) >= 1
            )

        elif route == "NAME_ADDRESS_NUMBER":

            valid_name = [
                token
                for token in name
                if name_freq[token] <= cap
            ]

            valid_address = [
                token
                for token in address
                if address_freq[token] <= cap
            ]

            valid_numbers = [
                number
                for number in numbers
                if number_freq[number] <= cap
            ]

            recovered_here = (
                len(valid_name) >= 1
                and
                len(valid_address) >= 1
                and
                len(valid_numbers) >= 1
            )

        else:

            raise ValueError(
                f"Unknown route: {route}"
            )

        if recovered_here:
            recovered += 1

    return recovered


# ============================================================================
# BUILD SAMPLE INDEXES
# ============================================================================

def build_sample_indexes(
    s1_records,
    candidate_records,
    name_freq,
    address_freq,
    number_freq,
    cap,
):

    # These indexes contain ONLY the 10k candidate
    # records from our diagnostic sample.
    #
    # They estimate candidate multiplicity under
    # each route, but they are NOT the real full
    # dataset candidate volume.

    indexes = {
        "NAME_NAME": {},
        "ADDRESS_ADDRESS": {},
        "ADDRESS_NUMBER": {},
        "NAME_ADDRESS": {},
        "NAME_NUMBER": {},
        "NAME_ADDRESS_NUMBER": {},
    }

    def add(index, key, entity_id):

        if key is None:
            return

        index.setdefault(
            key,
            set(),
        ).add(entity_id)

    # Candidate indexes
    for entity_id, rec in candidate_records.items():

        valid_names = [
            token
            for token in rec["name_tokens"]
            if name_freq[token] <= cap
        ]

        valid_addresses = [
            token
            for token in rec["address_tokens"]
            if address_freq[token] <= cap
        ]

        valid_numbers = [
            number
            for number in rec["numbers"]
            if number_freq[number] <= cap
        ]

        # --------------------------------------------------------------------
        # NAME + NAME
        # --------------------------------------------------------------------

        for i in range(
            len(valid_names)
        ):

            for j in range(
                i + 1,
                len(valid_names)
            ):

                key = tuple(
                    sorted(
                        (
                            valid_names[i],
                            valid_names[j],
                        )
                    )
                )

                add(
                    indexes["NAME_NAME"],
                    key,
                    entity_id,
                )

        # --------------------------------------------------------------------
        # ADDRESS + ADDRESS
        # --------------------------------------------------------------------

        for i in range(
            len(valid_addresses)
        ):

            for j in range(
                i + 1,
                len(valid_addresses)
            ):

                key = tuple(
                    sorted(
                        (
                            valid_addresses[i],
                            valid_addresses[j],
                        )
                    )
                )

                add(
                    indexes["ADDRESS_ADDRESS"],
                    key,
                    entity_id,
                )

        # --------------------------------------------------------------------
        # ADDRESS + NUMBER
        # --------------------------------------------------------------------

        for address_token in valid_addresses:

            for number in valid_numbers:

                key = (
                    address_token,
                    number,
                )

                add(
                    indexes["ADDRESS_NUMBER"],
                    key,
                    entity_id,
                )

        # --------------------------------------------------------------------
        # NAME + ADDRESS
        # --------------------------------------------------------------------

        for name_token in valid_names:

            for address_token in valid_addresses:

                key = (
                    name_token,
                    address_token,
                )

                add(
                    indexes["NAME_ADDRESS"],
                    key,
                    entity_id,
                )

        # --------------------------------------------------------------------
        # NAME + NUMBER
        # --------------------------------------------------------------------

        for name_token in valid_names:

            for number in valid_numbers:

                key = (
                    name_token,
                    number,
                )

                add(
                    indexes["NAME_NUMBER"],
                    key,
                    entity_id,
                )

        # --------------------------------------------------------------------
        # NAME + ADDRESS + NUMBER
        # --------------------------------------------------------------------

        for name_token in valid_names:

            for address_token in valid_addresses:

                for number in valid_numbers:

                    key = (
                        name_token,
                        address_token,
                        number,
                    )

                    add(
                        indexes["NAME_ADDRESS_NUMBER"],
                        key,
                        entity_id,
                    )

    return indexes


# ============================================================================
# CANDIDATE MULTIPLICITY ON THE SAMPLE
# ============================================================================

def estimate_candidate_counts(
    pair_data,
    indexes,
    s1_records,
    name_freq,
    address_freq,
    number_freq,
    cap,
):

    stats = {}

    for route, index in indexes.items():

        counts = []
        covered = 0

        for pair in pair_data:

            s1 = s1_records[
                pair["source1_entity_id"]
            ]

            valid_names = [
                token
                for token in s1["name_tokens"]
                if name_freq[token] <= cap
            ]

            valid_addresses = [
                token
                for token in s1["address_tokens"]
                if address_freq[token] <= cap
            ]

            valid_numbers = [
                number
                for number in s1["numbers"]
                if number_freq[number] <= cap
            ]

            candidate_ids = set()

            if route == "NAME_NAME":

                for i in range(
                    len(valid_names)
                ):

                    for j in range(
                        i + 1,
                        len(valid_names)
                    ):

                        key = tuple(
                            sorted(
                                (
                                    valid_names[i],
                                    valid_names[j],
                                )
                            )
                        )

                        candidate_ids.update(
                            index.get(
                                key,
                                set(),
                            )
                        )

            elif route == "ADDRESS_ADDRESS":

                for i in range(
                    len(valid_addresses)
                ):

                    for j in range(
                        i + 1,
                        len(valid_addresses)
                    ):

                        key = tuple(
                            sorted(
                                (
                                    valid_addresses[i],
                                    valid_addresses[j],
                                )
                            )
                        )

                        candidate_ids.update(
                            index.get(
                                key,
                                set(),
                            )
                        )

            elif route == "ADDRESS_NUMBER":

                for address_token in valid_addresses:

                    for number in valid_numbers:

                        candidate_ids.update(
                            index.get(
                                (
                                    address_token,
                                    number,
                                ),
                                set(),
                            )
                        )

            elif route == "NAME_ADDRESS":

                for name_token in valid_names:

                    for address_token in valid_addresses:

                        candidate_ids.update(
                            index.get(
                                (
                                    name_token,
                                    address_token,
                                ),
                                set(),
                            )
                        )

            elif route == "NAME_NUMBER":

                for name_token in valid_names:

                    for number in valid_numbers:

                        candidate_ids.update(
                            index.get(
                                (
                                    name_token,
                                    number,
                                ),
                                set(),
                            )
                        )

            elif route == "NAME_ADDRESS_NUMBER":

                for name_token in valid_names:

                    for address_token in valid_addresses:

                        for number in valid_numbers:

                            candidate_ids.update(
                                index.get(
                                    (
                                        name_token,
                                        address_token,
                                        number,
                                    ),
                                    set(),
                                )
                            )

            counts.append(
                len(candidate_ids)
            )

            if pair[
                "candidate_entity_id"
            ] in candidate_ids:

                covered += 1

        if counts:

            arr = np.array(
                counts,
                dtype=np.int64,
            )

            stats[route] = {
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

        else:

            stats[route] = {
                "median": 0,
                "p90": 0,
                "p95": 0,
                "p99": 0,
                "max": 0,
                "mean": 0,
                "covered": 0,
                "coverage_pct": 0,
            }

    return stats


# ============================================================================
# MAIN
# ============================================================================

def main():

    print("=" * 80)
    print("002C COMPOSITE ROUTE SIMULATION")
    print("=" * 80)

    # ------------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------------

    missed = load_missed()

    s1, candidates = load_records(
        missed
    )

    # ------------------------------------------------------------------------
    # Global frequencies
    # ------------------------------------------------------------------------

    (
        name_freq,
        address_freq,
        number_freq,
    ) = count_global_components()

    # ------------------------------------------------------------------------
    # Build components
    # ------------------------------------------------------------------------

    print(
        "\nBuilding S1 components..."
    )

    s1_records = build_components(
        s1
    )

    print(
        "Building candidate components..."
    )

    candidate_records = build_components(
        candidates
    )

    # ------------------------------------------------------------------------
    # True-pair overlaps
    # ------------------------------------------------------------------------

    pair_data = build_pair_data(
        missed,
        s1_records,
        candidate_records,
    )

    print(
        f"\nTrue pairs analyzed: "
        f"{len(pair_data):,}"
    )

    routes = [
        "NAME_NAME",
        "ADDRESS_ADDRESS",
        "ADDRESS_NUMBER",
        "NAME_ADDRESS",
        "NAME_NUMBER",
        "NAME_ADDRESS_NUMBER",
    ]

    all_results = []

    # ------------------------------------------------------------------------
    # Evaluate each frequency cap
    # ------------------------------------------------------------------------

    for cap in FREQ_CAPS:

        print("\n" + "=" * 80)
        print(
            f"FREQUENCY CAP = {cap:,}"
        )
        print("=" * 80)

        print(
            "\nBuilding sample indexes..."
        )

        indexes = build_sample_indexes(
            s1_records,
            candidate_records,
            name_freq,
            address_freq,
            number_freq,
            cap,
        )

        # ------------------------------------------------------------
        # TRUE PAIR RECOVERY
        # ------------------------------------------------------------

        for route in routes:

            recovered = evaluate_route(
                pair_data,
                route,
                name_freq,
                address_freq,
                number_freq,
                cap,
            )

            recovery_pct = (
                recovered /
                len(pair_data) *
                100
            )

            # --------------------------------------------------------
            # Candidate multiplicity
            # --------------------------------------------------------

            stats = estimate_candidate_counts(
                pair_data,
                indexes,
                s1_records,
                name_freq,
                address_freq,
                number_freq,
                cap,
            )[route]

            result = {
                "cap": cap,
                "route": route,
                "recovered": recovered,
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
                "sample_candidate_coverage":
                    stats["coverage_pct"],
            }

            all_results.append(
                result
            )

            print(
                f"{route:<24} "
                f"recovery={recovery_pct:6.2f}% "
                f"median={stats['median']:8.1f} "
                f"p95={stats['p95']:8.1f} "
                f"max={stats['max']:8d}"
            )

    # ------------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------------

    result_df = pd.DataFrame(
        all_results
    )

    output_path = (
        EXP_DIR /
        "002c_composite_route_results.tsv"
    )

    result_df.to_csv(
        output_path,
        sep="\t",
        index=False,
    )

    print("\n" + "=" * 80)
    print("RESULT FILE")
    print("=" * 80)

    print(output_path)

    # ------------------------------------------------------------------------
    # Compact summary
    # ------------------------------------------------------------------------

    print("\n" + "=" * 80)
    print("BEST RECOVERY BY ROUTE")
    print("=" * 80)

    for route in routes:

        subset = result_df[
            result_df["route"] == route
        ]

        best = subset.loc[
            subset["recovery_pct"].idxmax()
        ]

        print(
            f"{route:<24} "
            f"cap={int(best['cap']):>6,} "
            f"recovery={best['recovery_pct']:6.2f}% "
            f"median={best['median_candidates']:8.1f} "
            f"p95={best['p95_candidates']:8.1f}"
        )

    print(
        "\nSimulation complete."
    )


if __name__ == "__main__":
    main()
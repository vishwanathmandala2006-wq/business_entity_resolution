from pathlib import Path
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

SAMPLE_SIZE = 10_000


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
    "llc",
}


def normalize_unicode(value):
    if pd.isna(value):
        return ""

    value = str(value).lower().strip()
    value = unicodedata.normalize("NFKC", value)

    value = re.sub(r"[^\w\s]", " ", value, flags=re.UNICODE)
    value = re.sub(r"\s+", " ", value)

    return value.strip()


def alnum(value):
    return re.sub(r"[^\w]", "", normalize_unicode(value), flags=re.UNICODE)


def get_tokens(value):
    norm = normalize_unicode(value)

    if not norm:
        return []

    return norm.split()


def core_name(value):
    """
    Remove common legal/business suffixes and retain informative tokens.
    """
    toks = get_tokens(value)

    filtered = [
        t for t in toks
        if t not in LEGAL_SUFFIXES
    ]

    return " ".join(filtered)


def compact_name(value):
    return alnum(value)


def sorted_name(value):
    toks = get_tokens(value)

    informative = [
        t for t in toks
        if t not in LEGAL_SUFFIXES
    ]

    return " ".join(sorted(informative))


def name_token_set(value):
    return frozenset(get_tokens(value))


# ============================================================================
# ADDRESS COMPONENTS
# ============================================================================

def address_numbers(value):
    if pd.isna(value):
        return tuple()

    return tuple(sorted(set(re.findall(r"\d+", str(value)))))


def address_tokens(value):
    return get_tokens(value)


def address_compact(value):
    return alnum(value)


def informative_address_tokens(value):
    toks = address_tokens(value)

    return {
        t for t in toks
        if len(t) >= 4 and not t.isdigit()
    }


def first_number(value):
    nums = address_numbers(value)

    return nums[0] if nums else ""


# ============================================================================
# COMPONENT ROUTES
# ============================================================================

def build_routes(row):
    name = row.business_name
    address = row.business_address
    country = normalize_unicode(row.country)

    name_norm = normalize_unicode(name)
    name_core = core_name(name)
    name_compact = compact_name(name)
    name_sorted = sorted_name(name)

    addr_norm = normalize_unicode(address)
    addr_compact = address_compact(address)

    nums = address_numbers(address)
    addr_tokens = informative_address_tokens(address)

    routes = {}

    # ------------------------------------------------------------------------
    # NAME ROUTES
    # ------------------------------------------------------------------------

    routes["NAME_NORMALIZED"] = (
        country,
        name_norm,
    )

    routes["NAME_CORE"] = (
        country,
        name_core,
    )

    routes["NAME_COMPACT"] = (
        country,
        name_compact,
    )

    routes["NAME_SORTED"] = (
        country,
        name_sorted,
    )

    # ------------------------------------------------------------------------
    # ADDRESS ROUTES
    # ------------------------------------------------------------------------

    routes["ADDRESS_COMPACT"] = (
        country,
        addr_compact,
    )

    # All address numbers together
    if nums:
        routes["ADDRESS_NUMBERS"] = (
            country,
            nums,
        )
    else:
        routes["ADDRESS_NUMBERS"] = None

    # Number + each informative address token
    if nums and addr_tokens:
        routes["ADDRESS_NUM_TOKEN_SET"] = (
            country,
            nums,
            tuple(sorted(addr_tokens)),
        )
    else:
        routes["ADDRESS_NUM_TOKEN_SET"] = None

    # Number + individual informative token
    num = first_number(address)

    if num and addr_tokens:
        routes["ADDRESS_NUM_TOKEN_KEYS"] = [
            (
                country,
                num,
                token,
            )
            for token in addr_tokens
        ]
    else:
        routes["ADDRESS_NUM_TOKEN_KEYS"] = []

    # ------------------------------------------------------------------------
    # CROSS COMPONENT
    # ------------------------------------------------------------------------

    if num and name_core:
        routes["NAMECORE_NUM"] = (
            country,
            name_core,
            num,
        )
    else:
        routes["NAMECORE_NUM"] = None

    return routes


# ============================================================================
# LOAD MISSED PAIRS
# ============================================================================

def load_missed_pairs():

    missed = pd.read_csv(
        MISSED_PATH,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        nrows=SAMPLE_SIZE,
    )

    print(f"Missed pairs loaded: {len(missed):,}")

    return missed


# ============================================================================
# LOAD RECORDS
# ============================================================================

def load_required_records(missed):

    s1_ids = set(missed["source1_entity_id"])
    candidate_ids = set(missed["candidate_entity_id"])

    print(f"S1 records required: {len(s1_ids):,}")
    print(f"Candidate records required: {len(candidate_ids):,}")

    s1 = pd.read_csv(
        S1_PATH,
        sep="\t",
        dtype=str,
        keep_default_na=False,
    )

    s1 = s1[s1["entity_id"].isin(s1_ids)].copy()

    candidate_ids_remaining = candidate_ids.copy()

    frames = []

    for path, source_name in [
        (S2_PATH, "S2"),
        (S3_PATH, "S3"),
    ]:

        for chunk in pd.read_csv(
            path,
            sep="\t",
            dtype=str,
            keep_default_na=False,
            chunksize=250_000,
        ):

            found = chunk[
                chunk["entity_id"].isin(candidate_ids_remaining)
            ].copy()

            if len(found):
                found["source"] = source_name
                frames.append(found)

                candidate_ids_remaining -= set(found["entity_id"])

            if not candidate_ids_remaining:
                break

    candidates = pd.concat(frames, ignore_index=True)

    return s1, candidates


# ============================================================================
# ROUTE COMPARISON
# ============================================================================

def compare_routes(s1_row, candidate_row):

    r1 = build_routes(s1_row)
    r2 = build_routes(candidate_row)

    result = {}

    # ------------------------------------------------------------------------
    # Direct routes
    # ------------------------------------------------------------------------

    for route in [
        "NAME_NORMALIZED",
        "NAME_CORE",
        "NAME_COMPACT",
        "NAME_SORTED",
        "ADDRESS_COMPACT",
        "ADDRESS_NUMBERS",
        "ADDRESS_NUM_TOKEN_SET",
        "NAMECORE_NUM",
    ]:

        a = r1.get(route)
        b = r2.get(route)

        result[route] = (
            a is not None
            and b is not None
            and a == b
        )

    # ------------------------------------------------------------------------
    # NAME TOKEN INTERSECTION
    # ------------------------------------------------------------------------

    n1 = name_token_set(s1_row.business_name)
    n2 = name_token_set(candidate_row.business_name)

    result["NAME_TOKEN_INTERSECTION"] = len(n1 & n2) > 0

    # ------------------------------------------------------------------------
    # ADDRESS TOKEN INTERSECTION
    # ------------------------------------------------------------------------

    a1 = informative_address_tokens(s1_row.business_address)
    a2 = informative_address_tokens(candidate_row.business_address)

    result["ADDRESS_TOKEN_INTERSECTION"] = len(a1 & a2) > 0

    # ------------------------------------------------------------------------
    # ADDRESS NUMBER INTERSECTION
    # ------------------------------------------------------------------------

    nums1 = set(address_numbers(s1_row.business_address))
    nums2 = set(address_numbers(candidate_row.business_address))

    result["ADDRESS_NUMBER_INTERSECTION"] = len(nums1 & nums2) > 0

    # ------------------------------------------------------------------------
    # NUMBER + ADDRESS TOKEN INTERSECTION
    # ------------------------------------------------------------------------

    if nums1 & nums2 and a1 & a2:
        result["NUMBER_AND_ADDRESS_TOKEN"] = True
    else:
        result["NUMBER_AND_ADDRESS_TOKEN"] = False

    return result


# ============================================================================
# MAIN
# ============================================================================

def main():

    print("=" * 80)
    print("002C ROUTE RECOVERY SIMULATION")
    print("=" * 80)

    missed = load_missed_pairs()

    s1, candidates = load_required_records(missed)

    s1_map = {
        row.entity_id: row
        for row in s1.itertuples(index=False)
    }

    candidate_map = {
        row.entity_id: row
        for row in candidates.itertuples(index=False)
    }

    route_counts = {}

    pair_results = []

    print("\nTesting routes...")

    for idx, pair in enumerate(
        missed.itertuples(index=False),
        start=1
    ):

        s1_row = s1_map.get(pair.source1_entity_id)
        candidate_row = candidate_map.get(pair.candidate_entity_id)

        if s1_row is None or candidate_row is None:
            continue

        result = compare_routes(
            s1_row,
            candidate_row,
        )

        for route, matched in result.items():

            if matched:
                route_counts[route] = (
                    route_counts.get(route, 0) + 1
                )

        pair_results.append({
            "source1_entity_id": pair.source1_entity_id,
            "candidate_entity_id": pair.candidate_entity_id,
            **result,
        })

        if idx % 1000 == 0:
            print(f"Processed: {idx:,}")

    result_df = pd.DataFrame(pair_results)

    print("\n" + "=" * 80)
    print("ROUTE RECOVERY RESULTS")
    print("=" * 80)

    total = len(result_df)

    print(f"\nEvaluated pairs: {total:,}\n")

    ranking = []

    for route in sorted(route_counts):

        count = route_counts[route]

        recall = (
            count / total * 100
            if total
            else 0
        )

        ranking.append(
            (
                route,
                count,
                recall,
            )
        )

    ranking.sort(
        key=lambda x: x[1],
        reverse=True,
    )

    for route, count, recall in ranking:

        print(
            f"{route:<35}"
            f"{count:>8,} "
            f"({recall:6.2f}%)"
        )

    # ------------------------------------------------------------------------
    # Combined route recovery
    # ------------------------------------------------------------------------

    print("\n" + "=" * 80)
    print("COMBINED RECOVERY")
    print("=" * 80)

    route_columns = [
        "NAME_NORMALIZED",
        "NAME_CORE",
        "NAME_COMPACT",
        "NAME_SORTED",
        "ADDRESS_COMPACT",
        "ADDRESS_NUMBERS",
        "ADDRESS_NUM_TOKEN_SET",
        "NAMECORE_NUM",
        "NAME_TOKEN_INTERSECTION",
        "ADDRESS_TOKEN_INTERSECTION",
        "ADDRESS_NUMBER_INTERSECTION",
        "NUMBER_AND_ADDRESS_TOKEN",
    ]

    result_df["ANY_ROUTE"] = result_df[
        route_columns
    ].any(axis=1)

    result_df["NAME_ROUTE"] = result_df[
        [
            "NAME_NORMALIZED",
            "NAME_CORE",
            "NAME_COMPACT",
            "NAME_SORTED",
            "NAME_TOKEN_INTERSECTION",
        ]
    ].any(axis=1)

    result_df["ADDRESS_ROUTE"] = result_df[
        [
            "ADDRESS_COMPACT",
            "ADDRESS_NUMBERS",
            "ADDRESS_NUM_TOKEN_SET",
            "ADDRESS_TOKEN_INTERSECTION",
            "ADDRESS_NUMBER_INTERSECTION",
            "NUMBER_AND_ADDRESS_TOKEN",
        ]
    ].any(axis=1)

    print(
        f"ANY ROUTE:       "
        f"{result_df['ANY_ROUTE'].sum():,} / {total:,} "
        f"({result_df['ANY_ROUTE'].mean()*100:.2f}%)"
    )

    print(
        f"NAME ROUTES:     "
        f"{result_df['NAME_ROUTE'].sum():,} / {total:,} "
        f"({result_df['NAME_ROUTE'].mean()*100:.2f}%)"
    )

    print(
        f"ADDRESS ROUTES:  "
        f"{result_df['ADDRESS_ROUTE'].sum():,} / {total:,} "
        f"({result_df['ADDRESS_ROUTE'].mean()*100:.2f}%)"
    )

    # ------------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------------

    output_path = EXP_DIR / "route_recovery_002c.tsv"

    result_df.to_csv(
        output_path,
        sep="\t",
        index=False,
    )

    print(
        f"\nDetailed output: {output_path}"
    )

    print("\n002C route simulation complete.")


if __name__ == "__main__":
    main()
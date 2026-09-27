from pathlib import Path
from collections import Counter
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
CHUNK_SIZE = 250_000

CAPS = [
    10,
    25,
    50,
    100,
    250,
    500,
    1_000,
    2_500,
    5_000,
    10_000,
    25_000,
    50_000,
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


def compact(value):

    value = normalize_unicode(value)

    return re.sub(
        r"[^\w]",
        "",
        value,
        flags=re.UNICODE,
    )


def compact_char4(value):

    value = compact(value)

    if len(value) < 4:
        return set()

    return {
        value[i:i + 4]
        for i in range(len(value) - 3)
    }


# ============================================================================
# LOAD MISSED PAIRS
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
# GLOBAL CHAR4 FREQUENCIES
# ============================================================================

def count_char4_frequencies():

    name_counter = Counter()
    address_counter = Counter()

    print("\n" + "=" * 80)
    print("COUNTING GLOBAL COMPACT CHAR4 FREQUENCIES")
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

                name_counter.update(
                    compact_char4(value)
                )

            for value in chunk[
                "business_address"
            ]:

                address_counter.update(
                    compact_char4(value)
                )

    print(
        f"\nUnique name CHAR4 signatures: "
        f"{len(name_counter):,}"
    )

    print(
        f"Unique address CHAR4 signatures: "
        f"{len(address_counter):,}"
    )

    return (
        name_counter,
        address_counter,
    )


# ============================================================================
# ANALYZE TRUE PAIRS
# ============================================================================

def analyze_pairs(
    missed,
    s1,
    candidates,
    name_counter,
    address_counter,
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

    results = []

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

        # --------------------------------------------------------------------
        # NAME
        # --------------------------------------------------------------------

        s1_name = compact_char4(
            s1_row.business_name
        )

        candidate_name = compact_char4(
            candidate_row.business_name
        )

        shared_name = (
            s1_name &
            candidate_name
        )

        name_freqs = sorted(
            name_counter[x]
            for x in shared_name
        )

        # --------------------------------------------------------------------
        # ADDRESS
        # --------------------------------------------------------------------

        s1_address = compact_char4(
            s1_row.business_address
        )

        candidate_address = compact_char4(
            candidate_row.business_address
        )

        shared_address = (
            s1_address &
            candidate_address
        )

        address_freqs = sorted(
            address_counter[x]
            for x in shared_address
        )

        # --------------------------------------------------------------------
        # Store
        # --------------------------------------------------------------------

        results.append({
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

            "name_shared_count":
                len(shared_name),

            "address_shared_count":
                len(shared_address),

            "name_min_freq":
                name_freqs[0]
                if name_freqs
                else None,

            "name_second_min_freq":
                name_freqs[1]
                if len(name_freqs) >= 2
                else None,

            "name_third_min_freq":
                name_freqs[2]
                if len(name_freqs) >= 3
                else None,

            "name_max_freq":
                name_freqs[-1]
                if name_freqs
                else None,

            "address_min_freq":
                address_freqs[0]
                if address_freqs
                else None,

            "address_second_min_freq":
                address_freqs[1]
                if len(address_freqs) >= 2
                else None,

            "address_third_min_freq":
                address_freqs[2]
                if len(address_freqs) >= 3
                else None,

            "address_max_freq":
                address_freqs[-1]
                if address_freqs
                else None,
        })

    return pd.DataFrame(results)


# ============================================================================
# RECOVERY CURVES
# ============================================================================

def recovery_curve(
    df,
    freq_column,
):

    rows = []

    total = len(df)

    for cap in CAPS:

        recovered = (
            df[freq_column]
            .fillna(float("inf"))
            <= cap
        ).sum()

        rows.append({
            "cap": cap,
            "recovered": int(recovered),
            "recovery_pct":
                recovered / total * 100,
        })

    return pd.DataFrame(rows)


# ============================================================================
# COMBINED NAME + ADDRESS
# ============================================================================

def combined_recovery(df):

    rows = []

    total = len(df)

    for cap in CAPS:

        name_ok = (
            df["name_min_freq"]
            .fillna(float("inf"))
            <= cap
        )

        address_ok = (
            df["address_min_freq"]
            .fillna(float("inf"))
            <= cap
        )

        any_ok = (
            name_ok |
            address_ok
        )

        both_ok = (
            name_ok &
            address_ok
        )

        rows.append({
            "cap": cap,

            "name_recovered":
                int(name_ok.sum()),

            "name_pct":
                name_ok.mean() * 100,

            "address_recovered":
                int(address_ok.sum()),

            "address_pct":
                address_ok.mean() * 100,

            "any_recovered":
                int(any_ok.sum()),

            "any_pct":
                any_ok.mean() * 100,

            "both_recovered":
                int(both_ok.sum()),

            "both_pct":
                both_ok.mean() * 100,
        })

    return pd.DataFrame(rows)


# ============================================================================
# MULTIPLE-SIGNATURE RECOVERY
# ============================================================================

def multiple_signature_recovery(
    df,
    prefix,
):

    if prefix == "name":
        count_column = "name_shared_count"
        min_column = "name_min_freq"
        second_column = "name_second_min_freq"
        third_column = "name_third_min_freq"

    else:
        count_column = "address_shared_count"
        min_column = "address_min_freq"
        second_column = "address_second_min_freq"
        third_column = "address_third_min_freq"

    rows = []

    total = len(df)

    for cap in CAPS:

        one = (
            df[min_column]
            .fillna(float("inf"))
            <= cap
        )

        two = (
            df[second_column]
            .fillna(float("inf"))
            <= cap
        )

        three = (
            df[third_column]
            .fillna(float("inf"))
            <= cap
        )

        rows.append({
            "cap": cap,

            "one_signature":
                int(one.sum()),

            "one_pct":
                one.mean() * 100,

            "two_signatures":
                int(two.sum()),

            "two_pct":
                two.mean() * 100,

            "three_signatures":
                int(three.sum()),

            "three_pct":
                three.mean() * 100,
        })

    return pd.DataFrame(rows)


# ============================================================================
# MAIN
# ============================================================================

def main():

    print("=" * 80)
    print("002C CHAR4 INFORMATION ANALYSIS")
    print("=" * 80)

    # ------------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------------

    missed = load_missed()

    s1, candidates = load_records(
        missed
    )

    # ------------------------------------------------------------------------
    # Global frequency
    # ------------------------------------------------------------------------

    (
        name_counter,
        address_counter,
    ) = count_char4_frequencies()

    # ------------------------------------------------------------------------
    # Analyze true pairs
    # ------------------------------------------------------------------------

    df = analyze_pairs(
        missed,
        s1,
        candidates,
        name_counter,
        address_counter,
    )

    print(
        f"\nPairs analyzed: "
        f"{len(df):,}"
    )

    # ------------------------------------------------------------------------
    # Basic shared-signature statistics
    # ------------------------------------------------------------------------

    print("\n" + "=" * 80)
    print("SHARED CHAR4 STATISTICS")
    print("=" * 80)

    print(
        f"Name pairs sharing >=1 CHAR4: "
        f"{(df['name_shared_count'] > 0).sum():,} "
        f"({(df['name_shared_count'] > 0).mean()*100:.2f}%)"
    )

    print(
        f"Address pairs sharing >=1 CHAR4: "
        f"{(df['address_shared_count'] > 0).sum():,} "
        f"({(df['address_shared_count'] > 0).mean()*100:.2f}%)"
    )

    print(
        f"Name mean shared CHAR4: "
        f"{df['name_shared_count'].mean():.2f}"
    )

    print(
        f"Address mean shared CHAR4: "
        f"{df['address_shared_count'].mean():.2f}"
    )

    # ------------------------------------------------------------------------
    # Name curve
    # ------------------------------------------------------------------------

    name_curve = recovery_curve(
        df,
        "name_min_freq",
    )

    print("\n" + "=" * 80)
    print("RAREST SHARED NAME CHAR4")
    print("=" * 80)

    print(
        name_curve.to_string(
            index=False,
            formatters={
                "recovery_pct":
                    lambda x: f"{x:.2f}%"
            },
        )
    )

    # ------------------------------------------------------------------------
    # Address curve
    # ------------------------------------------------------------------------

    address_curve = recovery_curve(
        df,
        "address_min_freq",
    )

    print("\n" + "=" * 80)
    print("RAREST SHARED ADDRESS CHAR4")
    print("=" * 80)

    print(
        address_curve.to_string(
            index=False,
            formatters={
                "recovery_pct":
                    lambda x: f"{x:.2f}%"
            },
        )
    )

    # ------------------------------------------------------------------------
    # Combined
    # ------------------------------------------------------------------------

    combined = combined_recovery(
        df
    )

    print("\n" + "=" * 80)
    print("COMBINED RAREST-SIGNATURE RECOVERY")
    print("=" * 80)

    print(
        combined.to_string(
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
    # Multiple signature analysis
    # ------------------------------------------------------------------------

    name_multiple = (
        multiple_signature_recovery(
            df,
            "name",
        )
    )

    address_multiple = (
        multiple_signature_recovery(
            df,
            "address",
        )
    )

    print("\n" + "=" * 80)
    print("NAME: MULTIPLE RARE CHAR4 SIGNATURES")
    print("=" * 80)

    print(
        name_multiple.to_string(
            index=False,
            formatters={
                "one_pct":
                    lambda x: f"{x:.2f}%",

                "two_pct":
                    lambda x: f"{x:.2f}%",

                "three_pct":
                    lambda x: f"{x:.2f}%",
            },
        )
    )

    print("\n" + "=" * 80)
    print("ADDRESS: MULTIPLE RARE CHAR4 SIGNATURES")
    print("=" * 80)

    print(
        address_multiple.to_string(
            index=False,
            formatters={
                "one_pct":
                    lambda x: f"{x:.2f}%",

                "two_pct":
                    lambda x: f"{x:.2f}%",

                "three_pct":
                    lambda x: f"{x:.2f}%",
            },
        )
    )

    # ------------------------------------------------------------------------
    # Quantiles of minimum frequency
    # ------------------------------------------------------------------------

    print("\n" + "=" * 80)
    print("MINIMUM SHARED SIGNATURE FREQUENCY QUANTILES")
    print("=" * 80)

    for column, label in [
        (
            "name_min_freq",
            "NAME",
        ),
        (
            "address_min_freq",
            "ADDRESS",
        ),
    ]:

        values = (
            df[column]
            .dropna()
            .astype(float)
        )

        if len(values):

            quantiles = values.quantile(
                [
                    0.01,
                    0.05,
                    0.10,
                    0.25,
                    0.50,
                    0.75,
                    0.90,
                    0.95,
                    0.99,
                ]
            )

            print(
                f"\n{label}"
            )

            for q, value in quantiles.items():

                print(
                    f"  p{int(q*100):02d}: "
                    f"{value:,.0f}"
                )

    # ------------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------------

    df.to_csv(
        EXP_DIR /
        "002c_char_information_pairs.tsv",
        sep="\t",
        index=False,
    )

    name_curve.to_csv(
        EXP_DIR /
        "002c_char_information_name.tsv",
        sep="\t",
        index=False,
    )

    address_curve.to_csv(
        EXP_DIR /
        "002c_char_information_address.tsv",
        sep="\t",
        index=False,
    )

    combined.to_csv(
        EXP_DIR /
        "002c_char_information_combined.tsv",
        sep="\t",
        index=False,
    )

    name_multiple.to_csv(
        EXP_DIR /
        "002c_char_information_name_multiple.tsv",
        sep="\t",
        index=False,
    )

    address_multiple.to_csv(
        EXP_DIR /
        "002c_char_information_address_multiple.tsv",
        sep="\t",
        index=False,
    )

    print("\n" + "=" * 80)
    print("OUTPUT FILES")
    print("=" * 80)

    print(
        EXP_DIR /
        "002c_char_information_pairs.tsv"
    )

    print(
        EXP_DIR /
        "002c_char_information_name.tsv"
    )

    print(
        EXP_DIR /
        "002c_char_information_address.tsv"
    )

    print(
        EXP_DIR /
        "002c_char_information_combined.tsv"
    )

    print(
        EXP_DIR /
        "002c_char_information_name_multiple.tsv"
    )

    print(
        EXP_DIR /
        "002c_char_information_address_multiple.tsv"
    )

    print(
        "\n002C CHAR4 information analysis complete."
    )


if __name__ == "__main__":
    main()
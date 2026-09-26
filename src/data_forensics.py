from pathlib import Path
from collections import Counter
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent

TRAIN_DIR = BASE_DIR / "dataset" / "train"
TEST_DIR = BASE_DIR / "dataset" / "test"


FILES = {
    "train_source1": TRAIN_DIR / "train_source1.tsv",
    "train_source2": TRAIN_DIR / "train_source2.tsv",
    "train_source3": TRAIN_DIR / "train_source3.tsv",
    "test_source1": TEST_DIR / "test_source1.tsv",
    "test_source2": TEST_DIR / "test_source2.tsv",
    "test_source3": TEST_DIR / "test_source3.tsv",
}


SOURCE_COLUMNS = [
    "entity_id",
    "business_name",
    "business_address",
    "country"
]


def profile_source(name, path, chunksize=250_000):

    print("\n" + "=" * 80)
    print(f"PROFILE: {name}")
    print("=" * 80)

    total_rows = 0

    missing = Counter()
    countries = Counter()

    unique_ids = set()
    unique_names = set()
    unique_addresses = set()

    duplicate_name_counts = Counter()
    duplicate_address_counts = Counter()

    name_lengths = []
    address_lengths = []

    first_chunk = True

    for chunk in pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        chunksize=chunksize,
        keep_default_na=True
    ):

        total_rows += len(chunk)

        # --------------------------------------------------
        # Missing values
        # --------------------------------------------------

        for col in SOURCE_COLUMNS:
            missing[col] += chunk[col].isna().sum()

        # --------------------------------------------------
        # Country distribution
        # --------------------------------------------------

        countries.update(
            chunk["country"]
            .fillna("<MISSING>")
            .value_counts()
            .to_dict()
        )

        # --------------------------------------------------
        # Unique IDs
        # --------------------------------------------------

        unique_ids.update(
            chunk["entity_id"].dropna().unique()
        )

        # --------------------------------------------------
        # Names
        # --------------------------------------------------

        names = chunk["business_name"].dropna()

        unique_names.update(names.unique())

        duplicate_name_counts.update(names.tolist())

        name_lengths.extend(
            names.str.len().tolist()
        )

        # --------------------------------------------------
        # Addresses
        # --------------------------------------------------

        addresses = chunk["business_address"].dropna()

        unique_addresses.update(addresses.unique())

        duplicate_address_counts.update(addresses.tolist())

        address_lengths.extend(
            addresses.str.len().tolist()
        )

        if first_chunk:
            print("\nColumns:", chunk.columns.tolist())
            first_chunk = False

    # ------------------------------------------------------
    # Duplicate statistics
    # ------------------------------------------------------

    duplicate_names = sum(
        count > 1
        for count in duplicate_name_counts.values()
    )

    duplicate_addresses = sum(
        count > 1
        for count in duplicate_address_counts.values()
    )

    # ------------------------------------------------------
    # Print results
    # ------------------------------------------------------

    print("\nRows:", f"{total_rows:,}")
    print("Unique entity IDs:", f"{len(unique_ids):,}")

    print("\nMissing values:")

    for col in SOURCE_COLUMNS:
        count = missing[col]
        percentage = (
            count / total_rows * 100
            if total_rows
            else 0
        )

        print(
            f"  {col:20s}: "
            f"{count:,} ({percentage:.3f}%)"
        )

    print("\nCountries:")

    for country, count in countries.most_common():
        percentage = count / total_rows * 100

        print(
            f"  {country:20s}: "
            f"{count:,} ({percentage:.2f}%)"
        )

    print("\nUnique business names:",
          f"{len(unique_names):,}")

    print(
        "Names appearing more than once:",
        f"{duplicate_names:,}"
    )

    print("\nUnique addresses:",
          f"{len(unique_addresses):,}")

    print(
        "Addresses appearing more than once:",
        f"{duplicate_addresses:,}"
    )

    if name_lengths:

        print("\nBusiness-name length:")
        print(
            f"  Min: {min(name_lengths)}"
        )
        print(
            f"  Median: {pd.Series(name_lengths).median():.1f}"
        )
        print(
            f"  Mean: {pd.Series(name_lengths).mean():.1f}"
        )
        print(
            f"  Max: {max(name_lengths)}"
        )

    if address_lengths:

        print("\nBusiness-address length:")
        print(
            f"  Min: {min(address_lengths)}"
        )
        print(
            f"  Median: {pd.Series(address_lengths).median():.1f}"
        )
        print(
            f"  Mean: {pd.Series(address_lengths).mean():.1f}"
        )
        print(
            f"  Max: {max(address_lengths)}"
        )


def profile_ground_truth(path):

    print("\n" + "=" * 80)
    print("PROFILE: GROUND TRUTH")
    print("=" * 80)

    df = pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        keep_default_na=False
    )

    print("\nRows:", f"{len(df):,}")

    print(
        "Unique S1 IDs:",
        f"{df['source1_entity_id'].nunique():,}"
    )

    # Number of matched entities per S1
    match_counts = []

    for value in df["matched_entity_ids"]:

        if not value or pd.isna(value):
            match_counts.append(0)
        else:
            match_counts.append(
                len(
                    [
                        x
                        for x in value.split(",")
                        if x.strip()
                    ]
                )
            )

    counts = pd.Series(match_counts)

    print("\nMatch-count distribution:")

    distribution = counts.value_counts().sort_index()

    for n, count in distribution.items():

        percentage = count / len(df) * 100

        print(
            f"  {n:>3} matches: "
            f"{count:,} S1 entities "
            f"({percentage:.2f}%)"
        )

    print("\nSummary:")

    print(
        "  Singleton S1 entities:",
        f"{(counts == 0).sum():,}"
    )

    print(
        "  S1 with >=1 match:",
        f"{(counts > 0).sum():,}"
    )

    print(
        "  Maximum matches for one S1:",
        counts.max()
    )

    print(
        "  Average matches per S1:",
        f"{counts.mean():.3f}"
    )

    print(
        "  Median matches per S1:",
        f"{counts.median():.1f}"
    )


if __name__ == "__main__":

    # ------------------------------------------------------
    # Profile source files
    # ------------------------------------------------------

    for name, path in FILES.items():

        if not path.exists():

            print(
                f"\nWARNING: {name} not found:"
                f"\n{path}"
            )

            continue

        profile_source(name, path)

    # ------------------------------------------------------
    # Ground truth
    # ------------------------------------------------------

    ground_truth_path = (
        TRAIN_DIR / "train_ground_truth.tsv"
    )

    if ground_truth_path.exists():

        profile_ground_truth(
            ground_truth_path
        )

    print("\n" + "=" * 80)
    print("FORENSICS COMPLETE")
    print("=" * 80)
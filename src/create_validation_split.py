from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent

GT_PATH = (
    BASE_DIR
    / "dataset"
    / "train"
    / "train_ground_truth.tsv"
)

OUTPUT_DIR = (
    BASE_DIR
    / "experiments"
    / "001_baseline"
)


def main():

    print("=" * 80)
    print("CREATING S1-LEVEL VALIDATION SPLIT")
    print("=" * 80)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print("\nLoading ground truth...")

    gt = pd.read_csv(
        GT_PATH,
        sep="\t",
        dtype=str,
        keep_default_na=False
    )

    print(
        "Total S1 entities:",
        f"{len(gt):,}"
    )

    # Deterministic shuffle.
    # random_state ensures that we can reproduce
    # exactly the same validation split later.
    gt = gt.sample(
        frac=1.0,
        random_state=42
    ).reset_index(drop=True)

    split_index = int(
        len(gt) * 0.80
    )

    train_gt = gt.iloc[
        :split_index
    ].copy()

    validation_gt = gt.iloc[
        split_index:
    ].copy()

    train_path = (
        OUTPUT_DIR
        / "train_ground_truth.tsv"
    )

    validation_path = (
        OUTPUT_DIR
        / "validation_ground_truth.tsv"
    )

    train_gt.to_csv(
        train_path,
        sep="\t",
        index=False
    )

    validation_gt.to_csv(
        validation_path,
        sep="\t",
        index=False
    )

    print("\nSplit complete.")

    print(
        "\nDevelopment S1:",
        f"{len(train_gt):,}"
    )

    print(
        "Validation S1:",
        f"{len(validation_gt):,}"
    )

    print(
        "\nDevelopment percentage:",
        f"{len(train_gt) / len(gt) * 100:.2f}%"
    )

    print(
        "Validation percentage:",
        f"{len(validation_gt) / len(gt) * 100:.2f}%"
    )

    print("\nFiles created:")

    print(
        "  ",
        train_path
    )

    print(
        "  ",
        validation_path
    )

    print("\n" + "=" * 80)
    print("VALIDATION SPLIT READY")
    print("=" * 80)


if __name__ == "__main__":
    main()
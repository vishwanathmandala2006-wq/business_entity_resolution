from pathlib import Path
from collections import defaultdict, Counter
import re
import time
import unicodedata

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent
TRAIN_DIR = BASE_DIR / "dataset" / "train"
EXP_DIR = BASE_DIR / "experiments" / "001_baseline"

S1_PATH = TRAIN_DIR / "train_source1.tsv"
S2_PATH = TRAIN_DIR / "train_source2.tsv"
S3_PATH = TRAIN_DIR / "train_source3.tsv"
VALIDATION_GT_PATH = EXP_DIR / "validation_ground_truth.tsv"
OUTPUT_PATH = EXP_DIR / "002c_validation_candidates.tsv"

NAME_CAP = 5_000
ADDRESS_CAP = 10_000
CHUNK_SIZE = 250_000


def normalize(value):
    if pd.isna(value):
        return ""

    value = str(value).lower().strip()
    value = unicodedata.normalize("NFKC", value)
    value = re.sub(r"[^\w]", "", value, flags=re.UNICODE)
    return value.strip()


def compact_char4(value):
    compact = normalize(value)

    if len(compact) < 4:
        return set()

    return {
        compact[i:i + 4]
        for i in range(len(compact) - 3)
    }


def load_validation_s1():
    gt = pd.read_csv(
        VALIDATION_GT_PATH,
        sep="\t",
        dtype=str,
        keep_default_na=False,
    )

    ids = set(gt["source1_entity_id"])

    s1 = pd.read_csv(
        S1_PATH,
        sep="\t",
        dtype=str,
        keep_default_na=False,
    )

    s1 = s1[s1["entity_id"].isin(ids)].copy()
    print(f"Validation S1 records: {len(s1):,}")
    return s1


def count_relevant_char4(path, field, relevant_signatures):
    counter = Counter()

    for chunk in pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        usecols=["business_name", "business_address"],
        chunksize=CHUNK_SIZE,
    ):
        values = chunk[field]
        for value in values:
            signatures = compact_char4(value)
            counter.update(signatures & relevant_signatures)

    return counter


def build_validation_lookup(s1, name_counter, address_counter):
    name_lookup = defaultdict(set)
    address_lookup = defaultdict(set)

    for row in s1.itertuples(index=False):
        for sig in compact_char4(row.business_name):
            if name_counter.get(sig, 0) <= NAME_CAP:
                name_lookup[sig].add(row.entity_id)

        for sig in compact_char4(row.business_address):
            if address_counter.get(sig, 0) <= ADDRESS_CAP:
                address_lookup[sig].add(row.entity_id)

    print(f"Validation name lookup size: {len(name_lookup):,}")
    print(f"Validation address lookup size: {len(address_lookup):,}")
    return name_lookup, address_lookup


def load_true_pairs():
    gt = pd.read_csv(
        VALIDATION_GT_PATH,
        sep="\t",
        dtype=str,
        keep_default_na=False,
    )

    true_pairs = set()
    for row in gt.itertuples(index=False):
        if row.matched_entity_ids:
            for candidate_id in row.matched_entity_ids.split(","):
                true_pairs.add((row.source1_entity_id, candidate_id))

    return true_pairs


def generate_candidate_stats(
    source_path,
    source_name,
    name_lookup,
    address_lookup,
    true_pairs,
):
    candidate_counts = Counter()
    retrieved_pairs = set()
    candidate_total = 0

    print(f"\nGenerating {source_name} candidates...")

    for chunk in pd.read_csv(
        source_path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        usecols=["entity_id", "business_name", "business_address"],
        chunksize=CHUNK_SIZE,
    ):
        for row in chunk.itertuples(index=False):
            candidate_id = row.entity_id
            name_sigs = compact_char4(row.business_name)
            address_sigs = compact_char4(row.business_address)
            matched_s1 = set()

            for sig in name_sigs:
                if sig not in name_lookup:
                    continue
                matched_s1.update(name_lookup[sig])

            for sig in address_sigs:
                if sig not in address_lookup:
                    continue
                matched_s1.update(address_lookup[sig])

            candidate_total += len(matched_s1)
            for s1_id in matched_s1:
                candidate_counts[s1_id] += 1
                pair = (s1_id, candidate_id)
                if pair in true_pairs:
                    retrieved_pairs.add(pair)

    print(f"  {source_name} candidate pairs: {candidate_total:,}")
    return candidate_total, candidate_counts, retrieved_pairs


def candidate_distribution(df):
    if df.empty:
        return {
            "median": 0.0,
            "p90": 0.0,
            "p95": 0.0,
            "p99": 0.0,
            "max": 0,
        }

    counts = (
        df.groupby("source1_entity_id")["candidate_entity_id"]
        .nunique()
        .to_numpy(dtype=np.int64)
    )

    if len(counts) == 0:
        return {
            "median": 0.0,
            "p90": 0.0,
            "p95": 0.0,
            "p99": 0.0,
            "max": 0,
        }

    return {
        "median": float(np.median(counts)),
        "p90": float(np.percentile(counts, 90)),
        "p95": float(np.percentile(counts, 95)),
        "p99": float(np.percentile(counts, 99)),
        "max": int(counts.max()),
    }


def compare_with_002a(new_pairs, old_path):
    if not old_path.exists():
        return None

    old_pairs = set()
    for chunk in pd.read_csv(
        old_path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        usecols=["source1_entity_id", "candidate_entity_id"],
        chunksize=500_000,
    ):
        old_pairs.update(
            zip(
                chunk["source1_entity_id"],
                chunk["candidate_entity_id"],
            )
        )

    added = len(new_pairs - old_pairs)
    overlap = len(new_pairs & old_pairs)
    print(f"  002A overlap:     {overlap:,}")
    print(f"  new beyond 002A:  {added:,}")
    return added


def eval_pair_counts(df):
    gt = pd.read_csv(
        VALIDATION_GT_PATH,
        sep="\t",
        dtype=str,
        keep_default_na=False,
    )

    true_pairs = set()
    for _, row in gt.iterrows():
        matches = row["matched_entity_ids"]
        if matches:
            for candidate in matches.split(","):
                true_pairs.add((row["source1_entity_id"], candidate))

    candidate_pairs = set(
        zip(df["source1_entity_id"], df["candidate_entity_id"])
    )

    retrieved = len(true_pairs & candidate_pairs)
    missed = len(true_pairs - candidate_pairs)
    recall = retrieved / len(true_pairs)

    true_by_s1 = {}
    for s1, entity in true_pairs:
        true_by_s1.setdefault(s1, set()).add(entity)

    candidate_by_s1 = {}
    for s1, entity in candidate_pairs:
        candidate_by_s1.setdefault(s1, set()).add(entity)

    covered = 0
    for s1 in true_by_s1:
        if true_by_s1[s1] & candidate_by_s1.get(s1, set()):
            covered += 1

    coverage = covered / len(true_by_s1)

    return {
        "candidate_pairs": len(candidate_pairs),
        "s2_candidates": int((df["candidate_entity_id"].str.startswith("S2-")).sum()),
        "s3_candidates": int((df["candidate_entity_id"].str.startswith("S3-")).sum()),
        "true_pairs": len(true_pairs),
        "retrieved": retrieved,
        "missed": missed,
        "pair_recall": recall,
        "s1_coverage": coverage,
    }


def main():
    start = time.time()
    print("=" * 80)
    print("002C - VALIDATION-ONLY CHAR4 BLOCKER")
    print("=" * 80)

    s1 = load_validation_s1()

    validation_name_signatures = set()
    validation_address_signatures = set()
    for row in s1.itertuples(index=False):
        validation_name_signatures.update(compact_char4(row.business_name))
        validation_address_signatures.update(compact_char4(row.business_address))

    name_counter = Counter()
    address_counter = Counter()
    for path in (S1_PATH, S2_PATH, S3_PATH):
        name_counter.update(
            count_relevant_char4(path, "business_name", validation_name_signatures)
        )
        address_counter.update(
            count_relevant_char4(path, "business_address", validation_address_signatures)
        )

    print(f"\nName counter size: {len(name_counter):,}")
    print(f"Address counter size: {len(address_counter):,}")
    print(f"Name cap: {NAME_CAP:,}")
    print(f"Address cap: {ADDRESS_CAP:,}")

    name_lookup, address_lookup = build_validation_lookup(s1, name_counter, address_counter)
    true_pairs = load_true_pairs()

    s2_total, s2_counts, s2_retrieved = generate_candidate_stats(
        S2_PATH, "S2", name_lookup, address_lookup, true_pairs
    )
    s3_total, s3_counts, s3_retrieved = generate_candidate_stats(
        S3_PATH, "S3", name_lookup, address_lookup, true_pairs
    )

    candidate_counts = s2_counts + s3_counts
    retrieved_pairs = s2_retrieved | s3_retrieved
    candidate_total = s2_total + s3_total
    count_values = list(candidate_counts.values())
    stats = {
        "median": float(np.median(count_values)) if count_values else 0.0,
        "p90": float(np.percentile(count_values, 90)) if count_values else 0.0,
        "p95": float(np.percentile(count_values, 95)) if count_values else 0.0,
        "p99": float(np.percentile(count_values, 99)) if count_values else 0.0,
        "max": max(count_values, default=0),
    }

    print(f"\nCandidate pairs counted: {candidate_total:,}")
    print(f"Unique S2 candidate pairs: {s2_total:,}")
    print(f"Unique S3 candidate pairs: {s3_total:,}")
    print("\nCandidate distribution by S1:")
    print(f"  median: {stats['median']:.2f}")
    print(f"  p90:    {stats['p90']:.2f}")
    print(f"  p95:    {stats['p95']:.2f}")
    print(f"  p99:    {stats['p99']:.2f}")
    print(f"  max:    {stats['max']}")

    true_s1_ids = {s1_id for s1_id, _ in true_pairs}
    retrieved_s1_ids = {s1_id for s1_id, _ in retrieved_pairs}
    eval_stats = {
        "candidate_pairs": candidate_total,
        "s2_candidates": s2_total,
        "s3_candidates": s3_total,
        "true_pairs": len(true_pairs),
        "retrieved": len(retrieved_pairs),
        "missed": len(true_pairs - retrieved_pairs),
        "pair_recall": len(retrieved_pairs) / len(true_pairs),
        "s1_coverage": len(true_s1_ids & retrieved_s1_ids) / len(true_s1_ids),
    }
    print("\nValidation evaluation:")
    print(f"  candidate pairs:       {eval_stats['candidate_pairs']:,}")
    print(f"  S2 candidates:         {eval_stats['s2_candidates']:,}")
    print(f"  S3 candidates:         {eval_stats['s3_candidates']:,}")
    print(f"  true pairs:            {eval_stats['true_pairs']:,}")
    print(f"  retrieved:             {eval_stats['retrieved']:,}")
    print(f"  missed:                {eval_stats['missed']:,}")
    print(f"  pair recall:           {eval_stats['pair_recall']:.4%}")
    print(f"  S1 coverage:           {eval_stats['s1_coverage']:.4%}")

    elapsed = time.time() - start
    print(f"\nRuntime: {elapsed:.1f} seconds")
    print(f"Output: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()

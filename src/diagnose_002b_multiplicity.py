from pathlib import Path
import pandas as pd
import numpy as np

PATH = Path("experiments/001_baseline/approx_candidates.tsv")
CHUNK_SIZE = 500_000

counts = {}

for chunk in pd.read_csv(
    PATH,
    sep="\t",
    dtype=str,
    usecols=["source1_entity_id"],
    chunksize=CHUNK_SIZE,
):
    vc = chunk["source1_entity_id"].value_counts()

    for s1_id, count in vc.items():
        counts[s1_id] = counts.get(s1_id, 0) + int(count)

values = np.array(list(counts.values()), dtype=np.int64)

print("=" * 70)
print("002B CANDIDATE MULTIPLICITY")
print("=" * 70)
print(f"S1 entities with candidates: {len(values):,}")
print(f"Total candidate pairs:       {values.sum():,}")
print(f"Median:                      {np.median(values):.2f}")
print(f"P90:                         {np.percentile(values, 90):.2f}")
print(f"P95:                         {np.percentile(values, 95):.2f}")
print(f"P99:                         {np.percentile(values, 99):.2f}")
print(f"Max:                         {values.max():,}")

for threshold in [10, 25, 50, 100, 250, 500, 1000, 5000]:
    print(
        f"S1 with >{threshold:>4} candidates: "
        f"{(values > threshold).sum():,}"
    )

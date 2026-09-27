# 002C experiment plan

## Hypothesis

A compact CHAR4 blocker using separate rarity caps for name and address should recover most of the remaining 002B miss patterns while keeping candidate volume far below the 95M reference pool. The key idea is that address CHAR4 signatures are more informative than name CHAR4 signatures, so the system should not use a single global cap. We also expect that `2+` rare signatures should be a stronger precision signal than a lone rare signature, but a single extremely rare signature should still be allowed when it is highly distinctive.

## Evidence from previous experiments

- Baseline exact normalization: 4.49M candidates, 28.83% pair recall, 64.77% S1 coverage.
- Structural blocking (002A): 7.45M candidates, 53.78% pair recall, 85.42% S1 coverage.
- Approximate signature blocking (002B): 95.46M candidates, 99.01% S1 coverage, ~93.11% pair recall as high-recall reference.
- The 10k 002B-miss diagnostic sample shows that many misses are not exact-name failures:
  - name character similarity >= 0.80: 43.07%
  - address character similarity >= 0.50: 55.80%
  - combined `name >= 0.80 OR address >= 0.50`: 73.70%
- The CHAR4 diagnostic results are the strongest lead:
  - name compact CHAR4 at cap 10k: 58.83% recovery
  - address compact CHAR4 at cap 10k: 69.70% recovery
  - address CHAR4 at cap 10k: 68.36% recovery
  - combined any-route recovery at cap 10k: 88.31%
- Multiple rare signatures help reduce ambiguity:
  - name compact CHAR4, cap 10k: one signature 58.83%, two signatures 45.60%, three signatures 32.48%
  - address compact CHAR4, cap 10k: one signature 69.70%, two signatures 60.72%, three signatures 52.93%
- Separate caps are justified because name and address have different rarity distributions and different value as evidence.

## Selected CHAR4 routes

We will test a minimal candidate union based on:

- `NAME_COMPACT_CHAR4`
- `ADDRESS_COMPACT_CHAR4`
- configured with separate caps rather than a single global cap

We will allow the union of both routes, while recording whether a candidate was supported by a rare name signature, a rare address signature, or both.

## Proposed frequency thresholds

- `NAME_CAP = 5_000`
- `ADDRESS_CAP = 10_000`

Rationale:

- At name cap 5,000, the sample recovery is 49.99%; still useful without over-expanding the key space.
- At address cap 10,000, recovery is 69.70%; this is the strongest single-route signal.
- Combined `any` recovery at 5,000 / 10,000 is already 83.03% / 88.31% on the sample, which is high enough to test whether a validation-blocking candidate set remains manageable.
- A stricter name cap prevents generic compact name trigrams from exploding; a looser address cap keeps address evidence available for weak-name cases.

## Expected candidate-volume risks

- The union of all rare CHAR4 signatures can still generate a large candidate set because the same char signatures are relatively common in business naming and addresses.
- The main risk is that a low cap on address signatures produces too many generic address matches.
- To control this, the experiment will evaluate only the validation split and will explicitly record candidate multiplicity statistics per S1 before deciding whether to tighten or add support conditions.

## Validation method

Run the blocker only on the deterministic validation split:

- validation S1 IDs from `experiments/001_baseline/validation_ground_truth.tsv`
- global CHAR4 frequencies computed from all train-source records
- candidate generation restricted to `S2` and `S3`
- evaluate against the validation GT using the existing `evaluate_candidate_file.py` logic
- record per-S1 candidate distribution (median, p90, p95, p99, max)
- compare against 002A and the 002B reference pool as context, without treating 002B as a mandatory hard constraint

## Success criteria

The route is promising only if it meets all of the following:

1. pair recall materially exceeds 002A
2. S1 coverage materially exceeds 002A
3. candidate volume stays far below the approximate 95M 002B reference pool
4. the per-S1 candidate distribution is not pathological for a large subset of S1 entities
5. the new route creates a clear precision signal via evidence count and rarity-aware support

## Rollback criteria

Reject or tighten the route if any of the following occur:

- candidate count explodes beyond a practical range for downstream pairwise ML scoring
- pair recall remains close to 002A while candidate volume rises sharply
- a large fraction of S1 records receive extreme candidate multiplicities
- the route is dominated by one very common signature with poor precision

In that case, we tighten the cap, reintroduce a required `2+ rare signature` support rule, or abandon CHAR4 union in favor of a more selective hybrid route.

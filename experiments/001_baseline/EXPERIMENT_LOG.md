# EXPERIMENT LOG

## Experiment ID: 002C1

### Hypothesis

A combined compact CHAR4 blocker with separate name/address rarity caps
would recover a large fraction of the remaining 002B misses while staying
materially smaller than the 95M reference pool.

### Implementation

Validation-only candidate generation using rare compact CHAR4 signatures
from business names and business addresses, with:

- NAME_CAP = 5,000
- ADDRESS_CAP = 10,000

Global signature frequencies were counted across train S1/S2/S3.
Candidate generation was performed for the validation S1 set against S2/S3.

### Status

**INCONCLUSIVE / ABORTED**

### Reason

The end-to-end validation run terminated before reaching the final
evaluation stage. Therefore, no validated candidate count, pair recall,
S1 coverage, or candidate-distribution metrics were produced.

No unsupported performance claim is made for 002C1.

### Implementation Note

The current implementation performs global CHAR4 frequency counting
across S1/S2/S3 and subsequently rescans S2/S3 for candidate generation.

The script defines an `OUTPUT_PATH`, but the current implementation does
not materialize `002c_validation_candidates.tsv`.

### Decision

**Do not use 002C1 as the final candidate-generation blocker.**

The experiment is retained as an exploratory blocking experiment and its
diagnostic findings remain useful for downstream evidence/feature
engineering.

### Validated Reference: 002B

002B remains the validated high-recall candidate-generation reference:

- Candidate pairs: **95,461,677**
- True validation pairs: **1,527,797**
- Retrieved: **1,422,510**
- Missed: **105,287**
- Pair recall: **93.1086%**
- S1 coverage: **99.0114%**

### Downstream Use of 002C Findings

The earlier 002C diagnostic experiments indicate that compact CHAR4
representations can recover portions of the remaining 002B misses.

These representations may therefore be incorporated as downstream
pair-level evidence/features rather than being used as an independent
final blocking stage.

### Handoff

Blocking research is considered complete for Member A.

The following downstream tasks are handed to Member B:

1. Candidate compression
2. Candidate/evidence representation
3. Pair feature engineering
4. Hard-negative construction
5. TRACE-ER evidence features
6. ML matcher training
7. Hard-negative mining
8. F0.5 threshold calibration
9. S1-level decision layer
10. Zero/one/multiple match handling
11. Test inference
12. `candidate_pairs.tsv` generation
13. `matching_results.tsv` generation
14. Official output validation
15. Final methodology and result documentation
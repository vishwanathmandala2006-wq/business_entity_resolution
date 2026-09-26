\# Amazon ML Challenge 2026 — Business Entity Resolution



\## Project



Business entity resolution across three independent sources:



\- S1 — deduplicated reference entities

\- S2 — noisy business records

\- S3 — noisy business records



For every S1 entity, the system identifies zero, one, or multiple

matching S2/S3 records.



\## Architecture



TRACE-ER:



Trust-aware

Reliability-weighted

Adaptive

Contradiction-aware

Evidence-based Entity Resolution



Pipeline:



S1

→ Normalization

→ Candidate Generation

→ Candidate Compression

→ Pair Feature Engineering

→ ML Matching

→ Decision Layer

→ Final Matches



\## Current Candidate Generation



\### Experiment 001 — Exact Baseline



Pair recall: 28.8282%



S1 coverage: 64.7729%



\### Experiment 002A — Structural Blocking



Candidate pairs: 7,451,209



Pair recall: 53.7826%



S1 coverage: 85.4195%



Runtime: approximately 9h43m



\### Experiment 002B — Approximate Signature Blocking



Candidate pairs: 95,461,677



Pair recall: 93.1086%



S1 coverage: 99.0114%



Runtime: approximately 1h56m



\## Current Status



002B provides high candidate recall but produces a very large candidate

pool.



Next step:



1\. Analyze the 105,287 missed true pairs.

2\. Develop targeted additional blocking routes.

3\. Investigate candidate compression.

4\. Pass the manageable candidate set to the ML matching layer.



\## Team



Member A:

\- candidate generation

\- blocking

\- candidate recall



Member B:

\- pair feature engineering

\- ML matching

\- hard-negative mining

\- decision layer

\- precision optimization


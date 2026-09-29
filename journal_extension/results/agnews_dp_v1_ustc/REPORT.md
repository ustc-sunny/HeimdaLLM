# AG News DP v1 — USTC continuation

Status: **complete**, 12/12 complete runs.

This run restarts the unfinished GitHub DP plan on the USTC server. The released rental instance's DP outputs were not present in GitHub and are not included.

Privacy scope: record-level DP of the generated synthetic release. Downstream FL updates remain Non-DP. No end-to-end FL privacy claim is made.

Protocol: epsilon targets 8/4/2/1; delta 1e-5; seeds 57/58/59; 120 records per client; two independent client LoRA adapters; five generator epochs; 32 synthetic records per class; 50 FL rounds; evaluation before training and after every round; fixed 512-example development set.

All tasks share physical GPU 1. Runtime is not used for speed comparisons.

| Epsilon target | Completed seeds | Final accuracy (mean ± sample SD, %) | Final macro F1 |
|---:|---:|---:|---:|
| 8 | 3/3 | 39.84 ± 6.12 | 0.3527 |
| 4 | 3/3 | 39.71 ± 7.05 | 0.3597 |
| 2 | 3/3 | 43.10 ± 6.07 | 0.3903 |
| 1 | 3/3 | 40.23 ± 6.17 | 0.3642 |

Per-seed observations and achieved accountant bounds are in `summary.json`. Every available evaluation is in `curves.csv`.

Three seeds do not establish statistical significance. Comparisons with the GitHub Non-DP v3 are descriptive because the DP generator uses float32 and fresh private randomness.

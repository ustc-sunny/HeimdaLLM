# Stage 3: trace-matched control and fixed-cohort replication

Prospective protocol: `/mnt/heimdallm-stage3-20261022/PROTOCOL.json`, frozen on
2026-10-22 before inspecting new-cohort accuracy. Base repository commit:
`04cb83d5ebb5cfb05e1851629d2578855f2f3906`.

The delivered device is **RTX A6000 48 GB**, not the previously discussed A100.
All MPI roles are co-located on this rented GPU; no distributed speedup claim.

## Bounded matrix

- Original clients 1/21: `NO_GUIDANCE_TRACE_LEGACY` × FL seeds 57/58/59.
- Fixed new clients 100/101: `PUB_NLI`, `PRIVATE_ZERO`, `REAL_MATCHED` × the same seeds.
- Each endpoint: 50 rounds, 750 inner updates, 1,500 FD directions, 3,000 objective queries.
- Same frozen 128-candidate public pool, 64 guidance records, 120 real records/client,
  512-record development split. Official test data are not read.
- No new DP release, epsilon scan, generator change, client expansion, or accuracy-based rerun.

The normalized no-guidance legacy operator uses `z/sqrt(d)` and outer multiplier 1.
For fresh independent Gaussian directions its linear-limit mean is `g/d`.
The production legacy pipeline also retains history-based candidate selection
on some updates; there the relevant mean is `E[u u^T | history] g`, and the
fresh-isotropic formula cannot be asserted for the entire training run. This is
a nominal radius/trace-scale ablation, **not** a tuned unbiased isotropic ZOO baseline. Guided
core code is unchanged. Original raw-z Stage-2 results are preserved.

The new client IDs were fixed before label inspection and outcome evaluation.
The cohort audit checks feasibility, source separation, text duplicates, and
the frozen development split. It also verifies byte-identical reproduction of
historical PRIVATE_ZERO before applying the same selection pipeline to the new
cohort. This is replication within the same uniform AG News partition, not a
new domain or distribution-shift experiment.

## Preparation and execution

1. `prepare_source.py`: inspect hashes and patch only the new isolated checkout;
   cloud-free loading guard and RNG-preserving checkpoint capture. Preserve source diff.
2. `audit_operator.py`: analytic and actual operator checks, without training.
3. `prepare_cohort.py`: restricted cohort/guidance construction; no secret noise.
4. `run_endpoint.py --arm NO_GUIDANCE_TRACE_LEGACY --seed 57 --smoke --tag trace_single`.
5. `orchestrate.py`: await that smoke; reproduce old PUB_NLI seed57 for one round;
   run two concurrent bridge smokes, check seed57/58 against history, select concurrency
   using measured throughput and resource headroom. Then execute exactly 12 endpoints.
6. `after_queue.py`: wait for all endpoints, then replay production BP/GGD at actual
   checkpoints after rounds 0/10/25/50 of new-cohort PUB_NLI seed57. Same state and
   paired diagnostic RNG across PUB_NLI/PRIVATE_ZERO/REAL_MATCHED; reference is the
   64-record REAL_MATCHED cloud gradient, not full client population risk gradient.

All launchers refuse existing outputs. Failed artifacts are retained; failed
gates stop the queue. There is no automatic retry or selection for favorable accuracy.
Formal endpoints use the same frozen protocol irrespective of smoke accuracy.
The concurrent smoke is an operational benchmark including startup/cache overhead,
not a paper speedup result. At each launch, unrelated GPU processes block execution.

## Storage and privacy

- Persistent outputs: `/mnt/heimdallm-stage3-20261022`.
- Scratch: `/root/heimdallm-stage3-20261022/scratch`; logs/checkpoints copied to data disk.
- `report/metrics.json`: completed endpoint metrics and within-new-cohort paired differences.
- `report/STAGE3_REPORT.md`: incremental status, not a completion claim.
- `report/COMMON_CHECKPOINT_DIAGNOSTICS.json`: created only after actual diagnostics.
- `TRAINING_GATE.json`, `SOURCE_AUDIT.json`, `cohort/AUDIT.json`: provenance and gate evidence.
- `BLOCKED.json` / `DIAGNOSTICS_BLOCKED.json`: failures, when applicable.

Private source records, votes, checkpoints and replay caches remain restricted.
Training checkpoints and reported training/dev metrics have **no DP guarantee**.
No new noise artifact is created. Do not publish restricted artifacts or credentials
to Git. Code and aggregate metrics can be reviewed for a later authorized upload.
This stage does not edit the paper or automatically push to GitHub.

## Data-order correction discovered during preflight

The initial two-round bridge failed exact historical reproduction. Historical
`read_instance_from_h5` appended paired text/labels in thread completion order,
so cache reconstruction changed ordering despite identical input hashes and seed.
Its positional `guid` could also fail to identify the corresponding text. The
training loss uses text/label tensors, not that guid. Separately, shortening
`comm_round` changes the second-round LR schedule; bridge tests therefore use
one round, whose LR agrees with the 50-round prefix.

`repair_data_order.py` preserves all failed artifacts and records a prospective
`DATA_ORDER_AMENDMENT.json` before any formal endpoint. New-cohort loading now
follows frozen partition order. The original no-guidance control and bridge
use the matching historical PUB_NLI seed's exact caches, after checking cached
text/label multisets against source HDF5. This anchors the original control to
PUB_NLI; other old arms may include additional uncontrolled ordering randomness.
No old result has been silently replaced or counted as newly completed.

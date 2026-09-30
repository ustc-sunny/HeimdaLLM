# AG News same-pipeline DP noise bridge — locked plan

Date: 2026-09-30. Written before observing any new bridge result.

## Question and scope

Does adding Gaussian noise to the otherwise identical Poisson-sampled,
per-record-clipped client LoRA training path account for the large loss of
AG News guidance utility seen in DP v1? This is a controlled, two-client
diagnostic, not an 80-client KDD reproduction or end-to-end FL DP claim.

## Fixed paired conditions

- Source clients 1 and 21, 120 fixed train records each, and the existing
  byte-identical 512-record balanced development partition. Official test is
  untouched.
- Seeds 57, 58, 59; two conditions for each seed: `zero_noise` and `dp_eps8`.
  Run seed 57 first as an execution/accounting pilot, then the remaining seeds
  without changing the protocol if all checks pass.
- Both conditions call the same `dp_client_synthetic.py` per-example training
  loop with independent Poisson sampling, global L2 clip C=1, FP32
  DistilGPT2 LoRA r=8/alpha=16/dropout=.05, 5 epochs/150 optimizer steps,
  target batch size 4, AdamW LR=5e-4/WD=.01, max length 192.
- `zero_noise` omits Gaussian noise and has **no DP guarantee**. Its synthetic
  records and private training diagnostics remain local/server-only controls;
  only aggregate accuracy and non-private protocol metadata may be published.
- `dp_eps8` calibrates Gaussian noise for record-level epsilon <=8 at
  delta=1e-5, with fresh non-public sampling/noise entropy. Validate its
  accountant and release manifest before downstream training.
- Public category prompts, decoding parameters, 32 records/class (128 total),
  public-only filtering, and all downstream settings are shared. DistilBERT
  adapter FedFwd runs 50 rounds with two logical clients, alpha=.5, legacy
  guided estimator, LR=.01, and 3,000 client objective queries per condition.
- The conditions use separate fresh model/adapter instances. Random Poisson
  draws and DP noise are not forced to be identical across conditions; the
  comparison is paired by public model initialization and downstream seed,
  not a bitwise counterfactual.

## Readouts and interpretation

Preserve each pre-update/round-0..49 development trajectory, final accuracy,
macro F1/loss, generation counts, source and artifact hashes, accountant
parameters, and code revision. Report every seed and paired zero-minus-DP
differences. Compare older clipped and DP v1 results only descriptively.
Do not infer broad client or dataset generality, statistical significance,
or an end-to-end privacy guarantee from this bridge.

Archive every complete run with SHA-256 on the server and in a verified local
backup. Publish only a curated summary and curves to `HeimdaLLM+`; never
publish private staging, unnoised synthetic text, checkpoints, or credentials.

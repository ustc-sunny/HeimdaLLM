# AG News matched Non-DP ablation — 2026-09-29

This protocol is written before observing any new ablation utility outcome.
Existing v3 and DP v1 artifacts are immutable. No new DP grid is launched here.

## Stage 0: no-guidance update audit

The historical direction is `u=beta*z/sqrt(d)` for `alpha=1`, with Gaussian z.
Its covariance is `beta²/d I`. The historical estimator `(directional derivative)*u`
therefore has expected scale `beta²/d` times the smoothed gradient. For the
450,340-dimensional adapter at beta=1, this is an important scale confound.
The historical hybrid estimator is deliberately preserved; no claim of an
implementation bug or a successful corrected baseline is assumed from algebra.

An opt-in pure-isotropic estimator multiplies the *outer estimator* by `d/beta²`.
It does not change the finite-difference query radius, h=0.01, or query count.
It is rejected for alpha<1; the hybrid covariance is not isotropic.

On seed 57, run five five-round probes: historical alpha=1 at LR=.01, pure
Gaussian alpha=1 without scale correction at LR=.01, and the
corrected alpha=1 estimator at LR=.01/.001/.0001. Log actual parameter updates
locally. Select the corrected baseline LR by highest final dev accuracy, then
lowest final dev loss, then smallest LR. Report all probes and their extra
1,500 objective queries. Selection is exploratory development, not test evidence.
If any numerical/completion validation fails, stop and diagnose before formal runs.

Both isotropic modes bypass historical-gradient candidate selection, which changes
the direction distribution in the legacy path. The raw and corrected isotropic
LR=.01 controls thus isolate the scale change under the same Gaussian sampler.
The original four-probe deployment was stopped after its historical arm completed
and during its first corrected arm, before formal training. Its completed legacy
result and incomplete corrected logs are retained separately, not counted in the
replacement's evidence. The replacement has a distinct v2 run ID.

## Stage 1: matched generator controls

Five generator modes, each seeds 57/58/59, 50 FL rounds, plus the calibrated
no-guidance baseline at the same seeds: **18 formal runs**. All new trained
generators are **Non-DP**, including clipped/no-noise.

| Mode | Sampling / objective | Clipping | Noise |
|---|---|---|---|
| public | pretrained only; no LoRA; no private generator reads | none | none |
| ordinary | shuffled batch, mean target-token loss | none | none |
| fixed_example | same shuffled batches, mean per-record loss | none | none |
| poisson | independent Poisson q=1/30, mean per-record loss | none | none |
| clipped | same Poisson draws/dropout stream as poisson | per-record global L2 C=1 | none |

All generator settings match DP v1: FP32 DistilGPT2, fresh local LoRA r=8,
alpha=16/dropout=.05 per client, client IDs 1/21 with all 120 records each,
5 epochs/150 optimizer steps, target batch 4, AdamW LR=5e-4/WD=.01, max length192.
Public category prompt, decoding temperature=.8/top_p=.9/top_k=0/repetition1.05,
max new tokens80, fixed 32/class (128 total), global public deduplication and
minimum-length filtering are shared. No private exact-match filter runs anywhere.
Training RNG is publicly seeded only in these explicitly Non-DP controls;
the DP v1 generator's fresh private sampling/noise randomness is untouched.

Variable-length ordinary batch loss weights target tokens, whereas DP v1 averages
per-record losses. The fixed_example bridge separates this objective change from
the sampling change. Ordinary and fixed_example reuse permutations. Poisson and
clipped reuse sample draws and dropout streams. All arms reinitialize models.

Guided FL arms keep the historical estimator/alpha=.5 and LR=.01. Calibrated
no-guidance uses alpha=1 and its separately selected LR. This is a tuned reference,
not an ablation that changes only the generator. Both have 3,000 formal objective
queries per seed; baseline development consumes additional probe queries.

Use the byte-identical fixed AG News dev partition (512 balanced records) from
DP v1, DistilBERT adapter, max length64, batch8, 2 logical clients on 1 worker,
v_num=1/pool_size=1/retry0, evaluate before training and every round. Official
test remains untouched. Primary endpoint is round49 accuracy; macro F1/loss and
full curves are secondary. Three seeds do not establish significance or universality.

## Diagnosis and decision gates

Private debug: losses, per-record gradient norms, clipping counts, gradient-sum
norms and LoRA update norms. Store only under private_staging, outside archives
and GitHub summaries. They are not DP releases or privacy-safe tuning evidence.

- If ordinary FP32 fails to recover useful guidance, diagnose environment,
  precision, generation quality and training objectives before attributing noise.
- Compare ordinary→fixed_example→poisson→clipped to locate utility loss.
- Compare clipped with historical DP v1 descriptively. A future aligned noisy
  comparison requires a separate locked protocol/accounting, not this queue.
- Assess the calibrated baseline even if it eliminates an earlier large benefit.
- Only after examining this stage decide a bounded DP tuning experiment or
  multi-dataset/client expansion; do not automatically launch either.

## Persistence

All processes use shared physical GPU1; do not stop other users' processes.
Run sequentially. Freeze the reviewed Git revision before launch. Write commands,
protocol hashes, validated metrics and 51-point trajectories. Archive each complete
run with SHA256 on the server. Sync raw archives locally and only curated summary,
curves, public toy checks and reports to GitHub `HeimdaLLM+`. Private staging,
private gradient diagnostics, passwords and checkpoints are excluded from GitHub.

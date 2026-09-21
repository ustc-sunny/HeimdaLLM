# AG News Non-DP guidance validation v3: fixed plan

Date: 2026-09-21. Record this plan before viewing v3 downstream results.

## Questions

1. Does client-synthetic guidance retain an advantage through 50 FL rounds?
2. Does client-local generator LoRA add useful information beyond the public
   pretrained DistilGPT2 plus the same semantic category prompt?
3. Do aggregate gradients and the two-client gradient subspaces explain the
   relative utility of client, public, shuffled and real guidance?

No DP-SGD, clipping, noise or epsilon scan is permitted in v3.

## Fixed configuration

- Seeds 57, 58, 59; fixed source clients 1 and 21; both participate each round.
- Original AG News `uniform_client_1000` partition; 120 source records/client.
- Reserved real-control clients 800–807; development clients 900–908.
- Development evaluation: fixed balanced 512 records. Official test is unused.
- DistilBERT adapter task model; 50 FL rounds; evaluation before training and
  after every round, yielding rounds -1,0,...,49.
- Same v2 client generator: DistilGPT2 LoRA, 5 epochs, semantic category names,
  max training length 192, max 80 new tokens, 128 generated records, 32/class.
- Fixed downstream alpha: no-guidance 1.0; every guided arm 0.5. Fixed finite
  difference budget (`max_var_retries=0`) and 3000 client objective queries per
  50-round arm are expected (2 clients × 15 batches × 2 FD queries × 50 rounds).

Six paired arms per seed:

1. no guidance;
2. client-local LoRA synthetic;
3. pretrained-only public synthetic;
4. same-source real matched control;
5. held-out-client real matched control;
6. client synthetic with shuffled labels.

The public generator uses the same base model, category prompts, decoding,
generation length, quota, class balance and filters. It must construct no LoRA
and execute zero optimizer steps. Private records may only supply the existing
matched quota structure and exact-match release filter; they are not inputs to
model optimization. Record the number of exact-match rejections. If zero, the
filter did not alter the released public sample set.

## RQ1 readouts

Preserve every evaluation point. Report per seed and across-seed means for:

- accuracy, macro F1 and loss trajectories;
- final round 49;
- mean accuracy over rounds 0–49 (trajectory AUC divided by horizon);
- checkpoints 0, 4, 9, 24 and 49;
- mean paired advantage over early rounds 0–9 and late rounds 40–49;
- first round reaching 35%, 40%, 45% and 50% accuracy, when reached.

Interpret the observed shape without forcing a preferred conclusion. A positive
early gap with a near-zero late gap means acceleration only. A positive late
gap means persistence at this 50-round horizon. Mixed signs across seeds are
reported as instability. Three seeds do not justify a statistical-significance
claim or a final-convergence claim.

## RQ2 readouts

Compare client-synthetic minus public-synthetic per seed for the same curve and
final metrics. A positive difference supports an incremental contribution from
client LoRA under this configuration. A zero or negative difference does not
show that local LoRA adds useful domain information. Do not infer privacy.

## Gradient/subspace diagnostic

At the initial task-model checkpoint for each seed, measure:

- aggregate gradient cosine/alignment to source-real and development gradients;
- equal-norm one-step development loss change;
- fraction of source-real and development gradient energy projected into the
  rank-at-most-two span of the two per-client gradients;
- principal angles between client-synthetic, public-synthetic, shuffled and
  source-real two-client subspaces;
- corresponding per-client gradient alignment.

Use 16 records/class/client for source-real and both synthetic subspaces,
32/class for held-out real, and 64/class for development. Shuffled data use all
64 records/client because label shuffling need not preserve per-client balance.
These are offline checkpoint diagnostics, not federated convergence evidence.

## Execution and stopping

Prepare three independent seed directories, validate the public control, run
diagnostics, then run all six 50-round arms. Three seed jobs may share the A40;
do not report speed from concurrent execution. Validate schedules, initial
metrics, query budgets and all evaluation rounds. Preserve every arm regardless
of outcome, create a complete report and stop. Do not start any DP experiment.

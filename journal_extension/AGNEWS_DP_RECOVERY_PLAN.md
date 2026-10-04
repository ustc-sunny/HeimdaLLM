# AG News DP guidance recovery: diagnostic protocol

Date: 2026-10-04. Fixed before running the independent label-consistency classifier.

The paired noise bridge found 72.92% final development accuracy for zero-noise
generation and 37.57% for record-level DP generation at epsilon <= 8,
delta=1e-5 (three seeds, 50 FL rounds). Older epsilon 1/2/4/8 runs all
finished near 40%. The next step is to locate the failure before changing
the generator. This is a two-source-client diagnostic; it is not an 80-client
KDD reproduction or an end-to-end FL privacy result.

## Read-only quality audit

- Compare all 128 generated records from seeds 57/58/59 in each of three arms:
  bridge `zero_noise`, bridge `dp_eps8`, and matched-ablation `public`.
- Verify archive SHA-256 against the existing summaries before analysis.
- Report per-arm/seed word length, normalized duplicate count, corpus
  distinct-2, prompt-echo rate, and an independent classifier's agreement
  with the requested class. Keep texts and per-record predictions private.
- Use `textattack/distilbert-base-uncased-ag-news`, immutable revision
  `52ee64de95f38323f136c6f6b05e1af7c433417e`, only for local inference.
  First measure its accuracy on the fixed balanced 512-record AG News
  development partition and verify the canonical World/Sports/Business/SciTech
  label mapping. If dev accuracy is below 85%, do not interpret its synthetic
  label-agreement scores as evidence of class quality.
- Run this audit on CPU by default. It needs no FL training or exclusive GPU.
  If CPU inference is too slow, choose a GPU from *current* free memory and
  utilization, cap batch size, and record device and competing memory usage.

## Decision after the audit

If DP texts lose class agreement or converge toward the public generator,
test a single public-only class-consistency filter as postprocessing of the
already private `dp_eps8` generator. The initial audit found DP label
agreement 60.94% and public 62.24%; normalized DP/public
exact-text overlap is 65/69/72 out of 128 for seeds 57/58/59. These numbers
motivated the following filter protocol before any filtered FL outcome exists.

Use a separate, task-agnostic NLI selector:
`facebook/bart-large-mnli` revision
`d7645e127eaf1aefc7862fd59a17a5aa8558b8ce`, trained on public MNLI.
Hypothesis template: `This news article is about {}.`; candidate descriptions:
`world politics and international affairs`, `sports`, `business and finance`,
and `science and technology`. Rank by the requested class's entailment score
normalized across the four candidates. Validate its canonical label ordering
on the fixed dev set first; if dev accuracy is below 70%, do not run FL with
this selector. For each source client/class of 16 DP records, retain the top
8 (64 total). Use the same selection and budget for public-generator records
as a baseline. Also take a deterministic label/client-stratified random 8/16
DP records as a budget-matched control. No private-source record is inspected
by the selector; the releases remain local to the server and local backup.
The original epsilon of the DP generator is unchanged by this postprocessing.

Evaluate the selected records with the independent AG News classifier and
then use the historical guided FedFwd settings for a seed-57, 50-round pilot
of filtered DP, random DP, and filtered public. If the pilot fails validation,
stop and inspect rather than silently replace a run. If filtered DP improves
over both controls, complete the same three arms on seeds 58/59. If not, inspect
generator training dynamics and aggregate downstream gradient alignment
locally before altering the method. Do not launch a new epsilon sweep merely
to repeat the known 1/2/4/8 failure.

For the downstream pilot, use physical A40 GPU2 only while it has at least
14,000 MiB free and at most 20% utilization immediately before each run.
All three MPI ranks map to GPU2; record the snapshot per run. Do not switch
devices between paired arms. At planning time GPU2 had about 18.7 GiB free
while GPUs 0/1 were saturated. Shared usage may change; a failed guard stops
the queue without discarding completed runs.

Only after a corrected DP condition beats the public-generator baseline on
all three fixed seeds should the broader privacy-accuracy curve, client count,
and dataset extension be run. Keep the official test set untouched until the
method and hyperparameters are fixed. Do not publish zero-noise text,
per-record diagnostics, checkpoints, or private-source-derived metrics.

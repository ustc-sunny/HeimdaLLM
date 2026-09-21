# AG News Non-DP follow-up: fixed evaluation plan

Date: 2026-09-21. This plan is recorded before downstream v2 results.

Question: can automatically generated client data provide useful guidance
without DP clipping or noise after correcting the conditioning/context limits?

Only the generator prompt and context limits change from v1:

- Prompt: `Category: {label}\nNews article:\n`, with public raw-label names
  World, Sports, Business, Science and Technology. Stored labels stay unchanged.
- Generator training length: 192 (previously 64).
- Maximum generated tokens: 80 (previously 48).

Keep the same source clients (1,21), 120 records per client, 5 generator epochs,
LoRA/optimizer/sampling settings, 128 generated records (32/class), real controls,
512-example development set, task model, and 5 downstream rounds.
No manual relabeling or semantic filtering is applied to the training data.
Official test data remain unused.

Run seeds 57,58,59. Each seed repeats generation and all five downstream arms;
within each seed the initial task model and query budget are paired. Generator
adapter initialization remains fixed at 57 as in v1. This is three stochastic
pipeline repetitions, not three independent client populations.

Primary readout: final development accuracy, paired synthetic-minus-no-guidance
and synthetic-minus-shuffled differences for each seed and their mean/range.
Also report macro F1, loss and real-guidance controls. Consistent positive
accuracy differences across seeds support short-run feasibility; a difference
of only a few examples is not strong evidence. Mixed or absent gains are
inconclusive/negative evidence for this configuration. Do not claim final
convergence, statistical significance, privacy guarantees or a DP trade-off.

Inspect a fixed subset (first four generated records per class per seed) for
category mismatch/truncation. Treat this as a qualitative audit, not a blind
or full-set label-accuracy estimate. Do not use it to selectively remove records.

Execution: three jobs share one A40, with independent run/staging/cache paths.
No latency/throughput comparison is valid from these concurrent runs.

```bash
for seed in 57 58 59; do
  nohup bash journal_extension/run_matpool_agnews_nondp_v2.sh \
    --seeds "$seed" --run-id "matpool_agnews_nondp_v2_seed${seed}" \
    > "/root/heimdallm-work/logs/agnews-v2-seed${seed}.runner.log" 2>&1 < /dev/null &
done
```

Preserve v1, save all v2 arms regardless of outcome, run cross-arm validation,
and back up source/results. Do not launch a DP sweep on this evidence alone.

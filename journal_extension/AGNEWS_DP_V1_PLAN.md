# AG News DP v1 analysis plan

This plan is fixed before inspecting any DP utility outcome.  The earlier v3
results remain the Non-DP mechanism validation; this experiment measures the
privacy--utility trade-off of client-local DP LoRA generation.

## Mechanism and privacy unit

- Privacy unit: one training record within one selected client.
- Adjacency: add or remove one record.
- Each source client trains a fresh DistilGPT2 LoRA; adapters are never shared.
- Each DP step samples every client record independently with Poisson rate
  `q = 1 / ceil(N / B)`.
- Per-example LoRA gradients are clipped to global L2 norm `C = 1.0`.
- Gaussian noise with standard deviation `sigma * C` is added to the clipped
  gradient sum before division by the expected batch size.
- Opacus 1.4.0's RDP accountant calibrates and verifies epsilon at
  `delta = 1e-5`.
- Selected client partitions are checked to be disjoint.  The release therefore
  uses parallel composition across clients and reports the maximum client
  epsilon.

The target epsilon grid is `{1, 2, 4, 8}`.  Each client has fixed public
`N = 120`, target batch size 4, five epochs, 30 steps per epoch, and 150 DP
steps.  The generator uses float32 for all DP conditions.

The implementation uses seeded PyTorch pseudorandom generators to make the
research measurement repeatable.  The accountant assumes ideal Poisson and
Gaussian randomness; the run is not presented as a cryptographic deployment.

## Release boundary

Generation uses the fixed public AG News label table: World, Sports, Business,
and Science and Technology.  Every epsilon condition emits 32 examples per
class across clients.  Labels and quotas do not depend on observed client label
histograms.  The DP path does not run a private exact-text match filter, because
that would access private records after DP training outside the accounted
mechanism.

The public manifest may contain mechanism parameters, per-client accountant
outputs, artifact hashes, and generation counts.  It must not contain private
text, input label histograms, per-example losses, gradient norms, clipping
rates, or hashes of private staging files.

## Utility protocol

- Seeds: 57, 58, and 59.
- Downstream task model and FL settings match Non-DP v3.
- Training: 50 FL rounds with evaluation before training and after every round.
- Development set: the same fixed balanced 512-example AG News split used in
  v3; the official test set remains untouched.
- Primary endpoint: final accuracy at round 49.
- Secondary endpoints: final macro F1 and loss, best accuracy, area under the
  accuracy trajectory, and paired trajectories by seed.
- Comparators: the v3 client-synthetic Non-DP arm and public-generator arm.
  Comparisons against v3 are descriptive because the DP generator uses float32.

Report every seed and mean/std across three seeds.  Do not claim statistical
significance from three seeds.  Do not select epsilon using the official test
set.

## Execution stages

1. Deployment smoke: one seed, one client subset, reduced records/epochs and
   one FL round.  It validates execution and accounting only.
2. Full epsilon grid: run all four locked epsilon values with the protocol
   above.
3. Validate every release manifest before downstream interpretation.
4. Summarize the privacy--utility curve only after all locked conditions finish.

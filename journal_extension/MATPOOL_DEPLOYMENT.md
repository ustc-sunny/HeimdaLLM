# Single-A40 deployment (2026-09-21)

The original rental endpoint was instance-specific and may no longer exist.
Credentials are not stored in this repository.

Workspace: `/root/heimdallm-work`; code: `/root/heimdallm-work/HeimdaLLM`.
The GitHub branch is `HeimdaLLM+`, based on KDD commit
`f5880a1ec78df30e28c3c3ef9ff310aa9b39d945`. The experiment drivers live in
the separate `journal_extension/` directory. The branch also includes the
FedFwd/FedML integration changes needed by those drivers; the repository's
`main` branch remains unchanged. Use `setup_matpool_a40.sh` to recreate the
tested environments on a new rental instance.

## Environments

Both environments clone the platform's `myconda` environment, keeping Python
3.8.15 and PyTorch 1.13.1+cu116. KDD uses adapter-transformers 3.1.0 (reports
Transformers 4.21.3); generation uses Transformers 4.35.0, PEFT 0.6.1,
Accelerate 0.24.1 and Opacus 1.4.0. Do not combine these two requirements files.

```bash
conda create -y -n heimdallm-kdd --clone myconda
conda create -y -n heimdallm-ton --clone myconda
conda run -n heimdallm-kdd pip install -r journal_extension/requirements-kdd-matpool.txt
conda run -n heimdallm-ton pip install -r journal_extension/requirements-ton-matpool.txt
```

Full environment freezes and installation logs are in the server's `logs/`.
Three-rank Open MPI communication and a GPU Opacus optimizer/accountant step
passed. This infrastructure check does not validate DP-LoRA privacy guarantees.
The experiment launcher now recognizes Open MPI and MPICH hostfile formats.

## Data provenance and limits

The KDD README's Google Drive archive was downloaded and inspected. It contains
AG News, Yelp and Yahoo, **not SST-2**. The old server's SST-2 HDF5 files and client
partition are still needed to reproduce the original split.

For deployment validation only, `prepare_deployment_sst2.py` uses the official
Stanford `trainDevTestTrees_PTB.zip`, retaining non-neutral root sentences as the
repository's original loader does: 6,920 training and 1,821 test examples.
It generates a **new**, deterministic partition named `deployment_balanced100`:
100 clients with 32 examples per label; 520 remaining training records are
unused. Seed is 20260921. Source and output SHA-256 hashes are recorded in
`data/sst2_deployment/manifest.json`. This is not the original KDD partition.

## Deployment smoke experiment

```bash
cd /root/heimdallm-work/HeimdaLLM
bash journal_extension/run_matpool_smoke.sh --preflight-only
nohup bash journal_extension/run_matpool_smoke.sh \
  > /root/heimdallm-work/logs/sst2-smoke.runner.log 2>&1 < /dev/null &
```

The launcher uses DistilGPT2 generation and DistilBERT adapter training, seed
57, source clients 1 and 21, 5 rounds and 3 MPI ranks sharing GPU 0. It runs
no-cloud, synthetic, same-source-real-matched, reserved-real-matched and
shuffled-label arms. Evaluation uses 512 balanced examples from reserved
training clients 90–98; the official test set is not used for evaluation.

Output: `results/matpool_a40_sst2_deployment_smoke_20260921/`.
This small experiment verifies execution, accounting of query budgets and
data separation. It is not evidence of a final privacy–accuracy trade-off.

## AG News feasibility experiment

The current experiment uses the archive's original `agnews_data.h5` and
`agnews_partition.h5`, with partition method `uniform_client_1000`.
Source clients 1 and 21 each have 120 training records. Reserved real-control
clients are 800–807; development clients are 900–908. Development evaluation
uses 128 records per class (512 total); the official test set is not used.
The HDF5 label vocabulary is preserved exactly: `{"1":0,"4":1,"2":2,"3":3}`.
Category names correspond to raw labels, not these internal classifier indices.

```bash
cd /root/heimdallm-work/HeimdaLLM
bash journal_extension/run_matpool_agnews.sh --preflight-only
nohup bash journal_extension/run_matpool_agnews.sh \
  > /root/heimdallm-work/logs/agnews-feasibility.runner.log 2>&1 < /dev/null &
```

Each client trains an independent DistilGPT2 LoRA for 5 epochs. Generation
produces 32 examples per class across the two clients (128 total). The five
paired arms use seed 57, 5 rounds, and a common DistilBERT initialization.
Four-class evaluation includes macro F1 and the full confusion matrix.
Output: `results/matpool_agnews_nondp_feasibility_v1/`.

After generation, `run_matpool_agnews_alignment.sh` runs offline gradient
diagnostics with three task-model initialization seeds. It reuses the same
generated data; the three diagnostics are not independent end-to-end repeats.
They also resample real/development records and shuffled labels by seed.
Both launchers refuse to overwrite existing completed artifacts. Consult the
run's `REPORT.md` for findings; successful generation alone does not establish
downstream utility or differential privacy.

## Before formal DP experiments

The existing `dp_synthetic_smoke.py` is a prototype, not the formal experiment
entry point. It hard-codes OPT-style LoRA targets (`q_proj`, `v_proj`), derives
the released label list from input records, writes per-record loss/gradient
statistics, and reports a Gaussian composition indicator instead of integrating
the intended per-client sampling/accounting pipeline. Its `--clients` value is
metadata rather than independent client training. Do not use its output as a
validated privacy guarantee or publish its raw diagnostic records as DP output.
The formal pipeline needs separate client adapters, a public fixed label table,
compatible LoRA targets, validated sampling/clipping/noise/accounting, and
release metadata that does not expose unaccounted private statistics.

The new `dp_client_synthetic.py` implements that formal record-level pipeline:
fresh per-client adapters, public fixed labels and quotas, Poisson sampling,
per-example global clipping, Gaussian noise, and Opacus RDP accounting.  Its
release is checked by `validate_dp_release.py`.  The old prototype remains only
for historical context and must still not be used as evidence.

The server has about 350 GB of local disk but `/mnt` has only 5 GB of network
storage. Keep model caches on local disk and back up code, environment records
and essential results separately before releasing the instance.

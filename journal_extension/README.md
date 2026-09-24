# HeimdaLLM+ journal extension

This directory contains the journal-extension experiments on top of the KDD
HeimdaLLM code. The Git branch is `HeimdaLLM+`, based on commit
`f5880a1ec78df30e28c3c3ef9ff310aa9b39d945`. The repository's `main` branch is
untouched.

The experiment drivers and reports are isolated in `journal_extension/`. This
branch also carries the small set of FedFwd/FedML integration changes required
by the v3 runner: pre-update evaluation, fixed logical-client schedules,
separate cloud datasets for the six control arms, query accounting, and more
robust MPI shutdown and error handling.

## Current validated result

`results/agnews_nondp_v3/` contains the compact, reviewable outputs for the
AG News Non-DP v3 experiment:

- `REPORT.md`: full experiment report;
- `GPT_ANALYSIS_REPORT.md`: self-contained report intended for external review;
- `METRICS.md` and `summary.json`: final and paired statistics;
- `curves.csv`: all 918 evaluation records from 18 runs;
- `v3_training_curves.pdf` and `.png`: accuracy, macro-F1, loss, and paired-difference curves.

The three-seed raw logs, generated H5 files, and validation manifests are kept
in the local offline backup rather than Git. Model checkpoints, private staging
records, credentials, and server-specific logs are never committed.

The validated v3 result is Non-DP. It does not establish a privacy guarantee.

## Fresh Matpool A40 setup

The tested image was Ubuntu 20.04, Python 3.8, PyTorch 1.13.1, CUDA 11.6,
cuDNN 8, and NVCC. On a fresh instance, run:

```bash
git clone --branch 'HeimdaLLM+' --single-branch \
  https://github.com/ustc-sunny/HeimdaLLM.git \
  /root/heimdallm-work/HeimdaLLM

cd /root/heimdallm-work/HeimdaLLM
bash journal_extension/setup_matpool_a40.sh
```

The setup script creates two separate conda environments, downloads the public
FedNLP archive and the two pretrained models, and runs the v3 preflight check.
The two environments must remain separate because the KDD training stack uses
`adapter-transformers==3.1.0`, while the generator uses newer Transformers and
PEFT versions.

If Google Drive is unreachable, the script can restore the two AG News H5 files
from the persistent Matpool archive recorded in `HEIMDALLM_DATA_FALLBACK_ARCHIVE`.
If Hugging Face is unreachable, it downloads the same DistilBERT and DistilGPT2
repositories from ModelScope, then verifies that the required Transformers
files are complete.

The public FedNLP archive is downloaded from the FwdLLM link supplied for these
experiments. It contains AG News, Yelp, and Yahoo. It does not contain the
original SST-2 H5 partition.

## Reproduce AG News Non-DP v3

After setup:

```bash
cd /root/heimdallm-work/HeimdaLLM
mkdir -p /root/heimdallm-work/logs

bash journal_extension/run_matpool_agnews_nondp_v3.sh --preflight-only

nohup bash journal_extension/run_matpool_agnews_nondp_v3.sh \
  > /root/heimdallm-work/logs/agnews-nondp-v3.runner.log 2>&1 < /dev/null &
```

The launcher runs seeds 57, 58, and 59 with 50 FL rounds and six arms:
no guidance, client-LoRA synthetic, public synthetic, same-source real,
held-out real, and shuffled-label synthetic. It evaluates before training and
after every round. No DP mechanism is enabled.

For a new experiment, pass a new `--run-id`; the runner refuses to overwrite an
existing result directory. Use `--phase prepare` and `--phase train` when data
generation and downstream training need to be scheduled separately.

## Main files

- `run_matpool_agnews_nondp_v3.sh`: fixed v3 entry point.
- `run_sst2_nondp_feasibility.sh`: shared multi-arm runner.
- `non_dp_client_synthetic.py`: client-LoRA and pretrained-only generation.
- `validate_public_generator_control.py`: public-generator isolation audit.
- `validate_nondp_smoke.py`: per-arm and cross-arm accounting validation.
- `gradient_subspace_diagnostics.py`: initial-checkpoint gradient diagnostics.
- `summarize_agnews_nondp_v3.py`: three-seed trajectory summary.
- `plot_agnews_nondp_v3.py`: publication-style four-panel figure.
- `AGNEWS_NONDP_V3_PLAN.md`: analysis plan fixed before reading v3 outcomes.

See `MATPOOL_DEPLOYMENT.md` for additional deployment history and limitations.

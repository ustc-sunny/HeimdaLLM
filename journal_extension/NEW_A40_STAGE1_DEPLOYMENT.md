# Single-A40 continuation: what the Git branch contains

The `HeimdaLLM+` branch contains the current KDD/FedFwd code, journal-extension experiment drivers, historical **aggregate** reports and curves, the Stage-1 audit/handoff, the fixed 128-candidate **public** Qwen pool, and two public selections. See `results/agnews_guidance_stage1_20261014_c1b238c/STAGE1_REPORT.md` for the current 15-outcome matrix. No new Stage-1 downstream run or epsilon-4 release was completed at the time of this package.

## On the rented A40

The prior tested image was Ubuntu 20.04, CUDA 11.6, PyTorch 1.13.1. Verify usable GPU memory on the newly rented machine before launching the historical two-client pilot. Keep source and result directories on the persistent data disk. On the new machine, start with:

```bash
git clone --branch 'HeimdaLLM+' --single-branch https://github.com/ustc-sunny/HeimdaLLM.git /path/on/data-disk/HeimdaLLM
cd /path/on/data-disk/HeimdaLLM
git rev-parse HEAD
sha256sum journal_extension/frozen_agnews_qwen_pool_20261013/pool/candidates.jsonl
nvidia-smi --query-gpu=index,name,memory.free,utilization.gpu --format=csv
```

`journal_extension/setup_matpool_a40.sh` is the existing fresh-A40 setup for the older DistilGPT2 v3 workflow. It creates separate KDD and generator environments and downloads public FedNLP AG News. It has **not** been validated for the new single-A40 Qwen/NLI Stage-1 workflow; the frozen public pool means Qwen generation is unnecessary for E1. Adjust `HEIMDALLM_WORK_ROOT`, `HEIMDALLM_CONDA_ROOT`, and `HEIMDALLM_BASE_ENV` to the new image before using that setup. Preserve the two-environment separation. Do not run USTC launchers unchanged: some hardcode `/home/zzkevin/heimdallm-journal-20260928`, four GPU choices, historical result paths, and host-specific MPI maps.

## Exact protocol assets and checks

- Public candidate and selection JSONL files are tracked under `frozen_agnews_qwen_pool_20261013/`, with original manifests and SHA-256 values. No Qwen model download is needed merely to reuse this pool.
- The public FedNLP AG News archive must be downloaded on the new server. Historical `agnews_data.h5` SHA-256: `1474dd264d535b50ad2a7b1cf714bf11e182c2fe5b07f5e982feb31a9b114616`; historical source partition SHA-256: `fd6196bc2518457856365ec8f8207b1eb4b278d165a829320fc5b9fdd0a67f12`.
- The historical fixed two-client, 512-dev partition has SHA-256 `e16e8551a87ba702cc23b4d56d57b9762b77c4fea8bf341b96d5759be7996a6c`. This small H5 is **not** in public Git because it exposes exact client/dev membership. Restore it from the local checked raw archive through a restricted transfer when new server credentials are provided. Without it, a run is not strictly paired with archived seed-57 results.
- The original private-vote guidance and same-source real guidance are Non-DP artifacts. Public Git contains their aggregate metrics and curves, not raw selections or client data. Recreate or securely transfer them after validating the source data and partition.
- The frozen target is 2 logical clients (1 and 21), 120 source records/client, 64 guidance slots, 50 updates and 51 development evaluations, `alpha=0.5`, LR `0.01`, `fd_step=0.01`, legacy estimator, DistilBERT adapter/FedFwd. The Stage-1 report and `FROZEN_PROTOCOL.json` distinguish verified fields from pending single-step checks.

Once the data, models and restricted fixed partition are present, run the read-only readiness check. Point the Python options at the actual separate environments created on the new image; the older Matpool setup names them `heimdallm-kdd` and `heimdallm-ton`.

```bash
python3 journal_extension/preflight_single_gpu_stage1.py \
  --work-root /path/on/data-disk \
  --fixed-partition /path/on/data-disk/protocol/fixed_partition.h5 \
  --kdd-python /path/to/conda/envs/heimdallm-kdd/bin/python \
  --ton-python /path/to/conda/envs/heimdallm-ton/bin/python \
  --gpu-index 0
```

The checker reports `ready_for_e1=false` and exits 2 whenever an asset is missing, a hash differs, imports fail, or the GPU does not meet the existing 14000 MiB/20% gate. It never launches training and cannot replace the dynamic E0 update-operator audit.

## Run gate

Before starting any new training, verify actual GPU availability, model/tokenizer and data hashes, fixed partition, MPI mapping for **one** GPU, and the E0 single-step update operator. Do not silently change client count, guidance budget, candidate pool, learning rate, or source/dev split. Then complete missing E1 arms under distinct run IDs. The epsilon-4 private-vote DP selector is **not implemented/audited**; do not run E2 until the zero-noise path is byte-identical to `PRIVATE_ZERO` and the mechanism review passes. The old DP-LoRA workflow is a different experiment.

The code and aggregate results are downloadable from GitHub now, but the precise frozen Stage-1 experiment also needs the restricted fixed partition/private guidance artifacts and a single-A40 launcher adaptation. Raw archives stay in the original local `local_backups/` directory; `results/agnews_guidance_stage1_20261014_c1b238c/LOCAL_RAW_ARCHIVE_INDEX.json` records their hashes for later transfer. GitHub is not the sole backup of raw runs.

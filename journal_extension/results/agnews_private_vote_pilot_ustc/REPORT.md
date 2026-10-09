# AG News: public generator + private vote, Non-DP GPU2 pilot

Completed 2026-10-09 on physical GPU2 of the shared USTC A40 server. Both arms completed 50 federated rounds and all 51 planned evaluations (pre-update round `-1` and rounds `0`–`49`). This is a **one-seed feasibility pilot**, not a full Aug-PE implementation, a differential privacy result, or a reproduction of the KDD-scale setting.

## Question and protocol

Can guidance generated without client-local generator fine-tuning approach the accuracy of a matched real-data oracle? A public, pretrained Qwen2.5-3B-Instruct model generated 32 candidate AG News briefs per class using public semantic category prompts. No private training text entered those prompts. On each of two source clients (IDs 1 and 21), 120 local training records voted for same-class candidates by nearest-neighbor similarity using public all-MiniLM-L6-v2 embeddings. The top eight distinct candidates per client and class yielded 64 balanced synthetic guidance records (16 per class). This one-step vote had **no noise, clipping, or DP accounting**. Selected texts fit the downstream 64-token limit.

The control contains 64 real records from the same source clients, matched to the synthetic label sequence. Each arm used the same KDD FedFwd DistilBERT adapter training path, fixed data partition and 512-record AG News development set, `alpha=0.5`, learning rate `0.01`, seed `57`, two logical clients, and 50 rounds. The official AG News test set was not evaluated. The fixed protocol and hashes are in [protocol.json](protocol.json); exact per-round metrics are in [curves.csv](curves.csv) and [summary.json](summary.json).

## Results

| Guidance | Final dev accuracy | Final macro-F1 | Mean accuracy, rounds 40–49 |
|---|---:|---:|---:|
| Public Qwen + private vote | 75.00% (384/512) | 74.54% | 74.80% |
| Same-source real oracle | 72.07% (369/512) | 70.64% | 71.58% |

The paired final difference was **+2.93 percentage points** for synthetic guidance; the last-ten-round mean difference was **+3.22 points**. The synthetic arm started slower and overtook the real arm in this run:

| FL rounds | Synthetic mean accuracy | Real mean accuracy | Synthetic − real |
|---|---:|---:|---:|
| 0–9 | 45.08% | 52.85% | −7.77 pp |
| 10–19 | 63.52% | 67.73% | −4.22 pp |
| 20–29 | 71.11% | 70.59% | +0.53 pp |
| 30–39 | 73.75% | 71.11% | +2.64 pp |
| 40–49 | 74.80% | 71.58% | +3.22 pp |

An audit with `textattack/distilbert-base-uncased-ag-news` agreed with the requested class on 57/64 selected synthetic samples (89.06%). All 64 were within 64 downstream tokens. Six of the seven audit disagreements were Business samples predicted as Science/Technology. The classifier is an auxiliary quality proxy, not an unbiased estimate of label correctness or a privacy test. Aggregate audit data are in [quality.json](quality.json).

## Interpretation and limits

This run shows that public-model generation plus local private-data voting can give useful AG News guidance under this small protocol, without fine-tuning the generator. It does **not** show that the private vote added value over the same public generator with a private-data-free selection rule: that matched public-selection control has not yet been run. The real arm is a non-deployable accuracy oracle, not a privacy baseline. A single seed, two source clients, 64 guidance records, and a development set cannot establish statistical significance, generalization to the official test set, or equivalence to the KDD paper's full client setting. No privacy guarantee applies to the released synthetic records because vote selection used private data without DP. A later study needs matched public-selection and multi-seed controls before implementing and evaluating any DP mechanism.

Raw generated text, real records, cloud H5 files, and full logs remain outside GitHub in the local backup `local_backups/agnews_private_vote_pilot_20261009/`. The verified archive SHA-256 values are `6cc0686e9f582e4a732519b753f2f8f0ae4609d1dc9aa3d33c24a3ba4acad29d` (synthetic) and `d8093d31deb596dfec94d29b74d56f217e4513f7a359af970088819e18e7a362` (real). GitHub contains only code, protocol metadata, aggregate quality metrics, and accuracy curves.

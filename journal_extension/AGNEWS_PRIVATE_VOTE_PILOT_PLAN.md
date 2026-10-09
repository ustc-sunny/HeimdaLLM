# AG News training-free private-vote pilot (fixed before results)

Date: 2026-10-09. This is a one-step Aug-PE-inspired **Non-DP feasibility pilot**,
not an implementation or privacy claim for the full Aug-PE algorithm.

## Question and fixed protocol

Can a publicly pretrained instruction model, with local private-data voting but
no generator fine-tuning, produce useful cloud guidance for the unchanged KDD
FedFwd task-training path? Run one paired seed (57), two source clients (1, 21),
120 train records per client, 50 FL rounds, and the existing fixed 512-record
AG News development set. Keep the official test set untouched.

Generate 32 public-model candidate briefs per category with Qwen2.5-3B-Instruct.
No private text enters its prompts. On each client, embed its private train
records and the candidates with public all-MiniLM-L6-v2; each record votes for
the closest candidate within its category. Retain eight unique candidates per
client and category, giving 64 synthetic records. Keep records within the
downstream DistilBERT 64-token limit. This vote-based selection is **Non-DP**.

Create a same-source real-data oracle matched to the selected synthetic label
sequence and record count. For both arms, use the historical guided FedFwd
configuration: DistilBERT adapter, alpha 0.5, legacy estimator, learning rate
0.01, 50 rounds, evaluation before training and after each round. Use physical
GPU2 only if its preflight has at least 14 GiB free and at most 20% utilization.

## Interpretation

Report all 51 development evaluations and final accuracy, macro-F1, and
synthetic-minus-real difference. One seed and two clients measure feasibility;
they cannot establish KDD-scale equivalence or statistical significance.
Generation and matching happen entirely on the server. Raw private records,
selected records, and cloud H5 files stay in private server/local backups;
GitHub may receive only code and aggregate metrics/curves. No epsilon or DP
claim is attached to this pilot. If the Non-DP result is promising, a later
study must implement and validate the full DP private-evolution mechanism
before sweeping privacy budgets.

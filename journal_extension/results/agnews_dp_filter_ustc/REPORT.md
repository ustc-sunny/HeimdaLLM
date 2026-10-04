# AG News DP synthetic postprocessing filter

Status: 9/9 runs completed.

Only the existing DP synthetic release has record-level epsilon<=8, delta=1e-5; downstream FL is Non-DP.
Official test untouched; evaluation uses the fixed 512-record development set.

| Arm | n | Final accuracy (%) | Sample SD (pp) |
|---|---:|---:|---:|
| filtered_dp | 3 | 54.6224 | 7.5703 |
| random_dp | 3 | 32.6172 | 6.9412 |
| filtered_public | 3 | 55.4688 | 4.6997 |

The selector retains 8 of 16 samples per source client/class; each FL arm uses 64 samples.
Random DP and filtered public are matched-budget controls.
No significance or end-to-end DP claim is made.

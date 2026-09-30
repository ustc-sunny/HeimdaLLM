# AG News matched Non-DP ablations

Status: 17/18 runs completed.

All new generator training is Non-DP. Historical DP comparisons are descriptive.
Official test untouched; only the fixed 512-record dev set is evaluated.

| Arm | n | Final accuracy (%) | Sample SD (pp) |
|---|---:|---:|---:|
| clipped | 3 | 72.6562 | 0.0000 |
| fixed_example | 3 | 74.7396 | 1.4659 |
| no_guidance | 2 | 35.6445 | 1.7954 |
| ordinary | 3 | 71.6797 | 5.9594 |
| poisson | 3 | 70.5078 | 1.6688 |
| public | 3 | 41.6667 | 6.6128 |

Generator arms use the historical guided estimator and learning rate 0.01.
The formal no-guidance arm uses an explicitly labeled isotropic covariance correction
and a probe-selected learning rate; probe tuning queries are additional and reported.
No significance or end-to-end DP claim is made. Private debug logs are excluded.

# AG News matched Non-DP ablations

Status: 4/5 runs completed.

All new generator training is Non-DP. Historical DP comparisons are descriptive.
Official test untouched; only the fixed 512-record dev set is evaluated.

| Arm | n | Final accuracy (%) | Sample SD (pp) |
|---|---:|---:|---:|
| isotropic_lr0001 | 1 | 28.9062 | — |
| isotropic_lr001 | 1 | 24.2188 | — |
| isotropic_raw_lr001 | 1 | 27.7344 | — |
| legacy_lr001 | 1 | 27.7344 | — |

Generator arms use the historical guided estimator and learning rate 0.01.
The formal no-guidance arm uses an explicitly labeled isotropic covariance correction
and a probe-selected learning rate; probe tuning queries are additional and reported.
No significance or end-to-end DP claim is made. Private debug logs are excluded.

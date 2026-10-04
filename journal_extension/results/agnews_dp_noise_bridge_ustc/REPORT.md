# AG News same-pipeline DP noise bridge

Status: 6/6 runs completed.

Only dp_eps8 synthetic generation is record-level DP; zero_noise and all downstream FL updates are Non-DP.
Official test untouched; only the fixed 512-record dev set is evaluated.

| Arm | n | Final accuracy (%) | Sample SD (pp) |
|---|---:|---:|---:|
| dp_eps8 | 3 | 37.5651 | 4.7615 |
| zero_noise | 3 | 72.9167 | 0.8807 |

Generator arms use the historical guided estimator and learning rate 0.01.
No significance or end-to-end DP claim is made. Private debug logs are excluded.

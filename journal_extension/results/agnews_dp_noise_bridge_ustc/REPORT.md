# AG News same-pipeline DP noise bridge

Status: 0/6 runs completed.

Only dp_eps8 synthetic generation is record-level DP; zero_noise and all downstream FL updates are Non-DP.
Official test untouched; only the fixed 512-record dev set is evaluated.

| Arm | n | Final accuracy (%) | Sample SD (pp) |
|---|---:|---:|---:|

Generator arms use the historical guided estimator and learning rate 0.01.
No significance or end-to-end DP claim is made. Private debug logs are excluded.

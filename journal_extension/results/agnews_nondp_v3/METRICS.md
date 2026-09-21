# AG News Non-DP v3 metrics

Three seeds, fixed source/development clients, 50 rounds, Non-DP. This is not a statistical-significance, final-convergence or privacy result.

| Seed | No guidance | Client synthetic | Public synthetic | Same-source real | Held-out real | Shuffled |
|---|---:|---:|---:|---:|---:|---:|
| 57 | 27.7344% | 81.0547% | 42.5781% | 81.2500% | 76.9531% | 25.3906% |
| 58 | 36.5234% | 75.9766% | 36.5234% | 77.5391% | 79.8828% | 35.9375% |
| 59 | 21.8750% | 71.4844% | 38.0859% | 79.6875% | 79.6875% | 21.0938% |
| Mean | 28.7109% | 76.1719% | 39.0625% | 79.4922% | 78.8411% | 27.4740% |

| Comparison | Final mean (pp) | Rounds 0–49 mean (pp) | Early 0–9 (pp) | Late 40–49 (pp) |
|---|---:|---:|---:|---:|
| client_minus_no_cloud | +47.4609 | +37.3503 | +14.7461 | +47.3763 |
| client_minus_public | +37.1094 | +30.4232 | +13.6589 | +37.1094 |
| client_minus_shuffled | +48.6979 | +38.4297 | +15.4818 | +48.6784 |
| public_minus_no_cloud | +10.3516 | +6.9271 | +1.0872 | +10.2669 |

All 18 arms, 51 evaluations per arm and 3000 client objective queries per arm were validated.

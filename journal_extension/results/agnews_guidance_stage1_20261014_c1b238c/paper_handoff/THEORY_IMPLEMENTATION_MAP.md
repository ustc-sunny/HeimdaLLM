# Theory–implementation map

| Theoretical object | Current code path | Assessment |
|---|---|
| Trainable dimension `d` | Historical ZGR logs `n=450340`; `tc_transformer_trainer_distribute.py` counts requires-grad adapter parameters | Static/log match; names/hash and checkpoint pending |
| Local random vector `z` | Gaussian sampled in client trainer, scaled `β√(α/d)` | Current source uses `α=.5`, `β=1` |
| Cloud `V z_g` | `FedSgdCloudManager` sends normalized cloud BP direction; client `self.pert` | Actual `pool_size=1`, rank-1 direction. General multi-column V not mapped |
| `Σ` | Covariance implied by one random component plus one data-dependent fixed cloud direction | Does not justify a generic isotropic/V-subspace theorem without explicit assumptions |
| `h` and directional derivative | `fwdgrad_utils.calculate_jvp`, central difference `h=.01` | Static mapping; no live single-step numerical check |
| ZO gradient estimate | Client accumulates `jvp*z_h`, legacy outer multiplier 1 | Historical guided arms share this; corrected no-guidance arms cannot be mixed |
| Server update | `FedSgdServer.aggregate`, sum gradient tensors then divide by `training_num`, apply scheduled learning rate | Code comment says weighted, actual loop lacks explicit `n*g`; verify scale before claims |
| Guidance publication | `augpe_nondp_pilot.select_candidates` private votes and sequential disjoint top-8 | Non-DP only; DP implementation not mapped |
| Publication privacy ε,δ | Conditional ideal Laplace argument in `../PRIVATE_GUIDANCE_RELEASE_SPEC.md` | Implementation and numerical RNG not verified; actual release count 0 |
| Communication cost | Client manager sends full gradient tensor list | Scalar-loss-only transcript and full-FL DP claims do not map |

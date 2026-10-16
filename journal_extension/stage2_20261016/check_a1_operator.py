#!/usr/bin/env python3
"""CPU-only exact-quadratic check of the Stage-2 raw-ZOO source branch."""

import json
import math
from types import SimpleNamespace

import torch

from forward_training.tc_transformer_trainer_distribute import ForwardTextClassificationTrainer
from forward_training.utils.fwdgrad_utils import calculate_jvp, zo_estimator_multiplier


def main():
    torch.manual_seed(57)
    d = 1000
    probe = object.__new__(ForwardTextClassificationTrainer)
    probe.trainable_params_number = d
    probe._component_norms_logged = False
    probe.pert = [torch.full((d,), float("nan"))]
    args = SimpleNamespace(alpha=1.0, beta=1.0, zo_estimator="isotropic_raw")
    z = torch.randn(d, dtype=torch.float64)
    result = probe._make_final_perturbation((z,), args)[0]
    if not torch.equal(result, z):
        raise AssertionError("pure-ZOO branch must use the unnormalized raw z")
    x = torch.randn(d, dtype=torch.float64)
    target = torch.randn(d, dtype=torch.float64)
    func = lambda params: 0.5 * ((params[0] - target) ** 2).sum()
    mean_loss, jvp, delta = calculate_jvp(func, (x,), (result,), h=0.01)
    exact = ((x - target) * z).sum()
    relative = abs((jvp - exact).item()) / max(1.0, abs(exact.item()))
    if relative > 1e-9 or not math.isclose(delta.item(), 0.02 * exact.item(), rel_tol=1e-9):
        raise AssertionError("central finite difference does not match exact directional derivative")
    scale = zo_estimator_multiplier("isotropic_raw", 1.0, 1.0, d)
    if scale != 1.0:
        raise AssertionError("raw Gaussian estimator must have no dimension correction")
    update = 0.01 * scale * jvp * z
    if not torch.isfinite(update).all():
        raise AssertionError("nonfinite update")
    print(json.dumps({"dimension": d, "raw_z_norm": z.norm().item(),
                      "sqrt_dimension": math.sqrt(d),
                      "fd_delta": delta.item(), "fd_directional_derivative": jvp.item(),
                      "exact_directional_derivative": exact.item(),
                      "relative_error": relative, "update_l2": update.norm().item(),
                      "cloud_nan_ignored": True, "status": "pass"}, indent=2))


if __name__ == "__main__":
    main()

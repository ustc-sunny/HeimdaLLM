#!/usr/bin/env python3
"""Check estimator covariance and training controls on public toy data only."""
import copy
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from forward_training.utils.fwdgrad_utils import calculate_jvp, zo_estimator_multiplier
from matched_nondp_synthetic import train_control


class ToyTokenizer:
    pad_token_id = 0
    eos_token_id = 1

    def encode(self, value, add_special_tokens=False):
        return [2] if value.startswith("Category:") else [3, 4, 5]


class ToyLM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(8, 4)
        self.head = torch.nn.Linear(4, 8)
        self.config = SimpleNamespace(use_cache=False)

    def forward(self, input_ids, attention_mask, labels):
        logits = self.head(self.embedding(input_ids))
        loss = torch.nn.functional.cross_entropy(
            logits[:, :-1].reshape(-1, 8), labels[:, 1:].reshape(-1), ignore_index=-100)
        return SimpleNamespace(loss=loss)


def main():
    torch.manual_seed(123)
    dimension, beta, draws = 32, 0.7, 100000
    gradient = torch.linspace(0.2, 1.0, dimension)
    z = torch.randn(draws, dimension) * beta / dimension ** 0.5
    estimate = ((z @ gradient).unsqueeze(1) * z).mean(0)
    corrected = estimate * zo_estimator_multiplier("isotropic", 1.0, beta, dimension)
    relative_error = (corrected - gradient).norm().item() / gradient.norm().item()
    assert relative_error < 0.035
    assert zo_estimator_multiplier("legacy", 0.5, 1.0, dimension) == 1.0
    try:
        zo_estimator_multiplier("isotropic", 0.5, beta, dimension)
    except ValueError:
        pass
    else:
        raise AssertionError("hybrid correction must be rejected")
    theta = torch.linspace(-0.2, 0.3, dimension)
    direction = z[0]
    _, derivative, _ = calculate_jvp(
        lambda p: 0.5 * (p[0] ** 2).sum(), (theta,), (direction,), h=0.01)
    assert torch.allclose(derivative, theta @ direction, atol=2e-6, rtol=1e-4)
    rows = [{"label": "1", "text": "public toy record"} for _ in range(8)]
    args = SimpleNamespace(max_length=12, prompt_template="Category: {label}\n",
        label_names={"1": "World"}, batch_size=4, epochs=2,
        learning_rate=0.001, weight_decay=0.01, max_grad_norm=1e9)
    base = ToyLM()
    results, diagnostics = {}, {}
    for mode in ("ordinary", "fixed_example", "poisson", "clipped"):
        args.mode = mode
        model = copy.deepcopy(base)
        _, _, diag = train_control(torch, model, ToyTokenizer(), rows, args,
                                   torch.device("cpu"), 77)
        results[mode] = torch.cat([p.detach().flatten() for p in model.parameters()])
        diagnostics[mode] = diag
    assert torch.allclose(results["ordinary"], results["fixed_example"], atol=2e-7, rtol=1e-6)
    assert torch.equal(results["poisson"], results["clipped"])
    assert diagnostics["clipped"]["clipped_examples"] == 0
    args.max_grad_norm, args.mode = 0.01, "clipped"
    _, _, clipped = train_control(torch, copy.deepcopy(base), ToyTokenizer(), rows,
                                  args, torch.device("cpu"), 77)
    assert clipped["clipped_examples"] > 0
    print(json.dumps({"status": "passed", "public_toy_data_only": True,
        "isotropic_covariance_relative_error": relative_error,
        "finite_difference_matches_quadratic_directional_derivative": True,
        "equal_length_token_mean_matches_record_mean": True,
        "inactive_clipping_matches_poisson_bitwise": True,
        "active_clipping_detected": True}))


if __name__ == "__main__":
    main()

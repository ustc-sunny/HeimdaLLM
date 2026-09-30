#!/usr/bin/env python3
"""Public toy check: zero-noise mode is Non-DP and DP accounting stays active."""
import copy
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import torch

from check_matched_ablation import ToyLM, ToyTokenizer
from dp_client_synthetic import dp_train_adapter
import validate_dp_release


class ToyAccountant:
    def __init__(self):
        self.steps = 0

    def step(self, *, noise_multiplier, sample_rate):
        assert noise_multiplier > 0 and 0 < sample_rate < 1
        self.steps += 1

    def get_privacy_spent(self, *, delta):
        assert self.steps == 2 and delta == 1e-5
        return 1.0, 4.0


def main():
    rows = [{"label": "1", "text": "public toy record %d" % i} for i in range(8)]
    options = dict(records_per_client=8, max_length=12, batch_size=4, epochs=1,
                   learning_rate=0.001, weight_decay=0.01, max_grad_norm=1.0,
                   delta=1e-5, epsilon_tolerance=0.01,
                   prompt_template="Category: {label}\n", label_names={"1": "World"},
                   target_epsilon=None, noise_multiplier=None)
    torch.manual_seed(23)
    base = ToyLM()
    args = SimpleNamespace(**options, diagnostic_zero_noise=True)
    _, control = dp_train_adapter(torch, copy.deepcopy(base), ToyTokenizer(), rows,
                                  args, torch.device("cpu"), 23, args.prompt_template,
                                  ToyAccountant, lambda **_: 0.7)
    assert control["noise_multiplier"] == 0.0
    assert control["epsilon"] is None and control["accountant"] == "none_non_dp_control"
    assert control["optimizer_steps"] == 2
    args = SimpleNamespace(**{**options, "target_epsilon": 8.0}, diagnostic_zero_noise=False)
    _, private = dp_train_adapter(torch, copy.deepcopy(base), ToyTokenizer(), rows,
                                  args, torch.device("cpu"), 23, args.prompt_template,
                                  ToyAccountant, lambda **_: 0.7)
    assert private["noise_multiplier"] == 0.7
    assert private["epsilon"] == 1.0 and private["accountant"] == "opacus_rdp"
    with tempfile.TemporaryDirectory() as directory:
        manifest = Path(directory) / "control.json"
        manifest.write_text(json.dumps({"status": "complete", "is_record_level_dp": False}))
        rejected = subprocess.run([sys.executable, str(Path(validate_dp_release.__file__)),
                                   "--manifest", str(manifest)], capture_output=True, text=True)
        assert rejected.returncode != 0
        assert "does not declare record-level DP" in rejected.stderr
    print(json.dumps({"status": "passed", "public_toy_data_only": True,
                      "zero_noise_has_no_accountant": True,
                      "positive_noise_keeps_accountant": True,
                      "zero_noise_rejected_by_dp_validator": True}))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Apply the preregistered paper-equivalent pure-ZOO operator in an isolated clone."""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path


COMMIT = "ad325aa320789c4f59f911898033844d5b19ccec"


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise RuntimeError("expected exactly one source anchor, found %d" % text.count(old))
    return text.replace(old, new)


def digest_bytes(data):
    return hashlib.sha256(data).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("repo", type=Path)
    args = parser.parse_args()
    repo = args.repo.resolve()
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    changes = subprocess.check_output(["git", "diff", "--name-only", "HEAD"],
                                      cwd=repo, text=True)
    if head != COMMIT or changes:
        raise RuntimeError("A1 clone must start clean at preregistered source commit")

    trainer = repo / "forward_training/tc_transformer_trainer_distribute.py"
    main_file = repo / "experiments/distributed/transformer_exps/run_tc_exps/fedavg_main_tc.py"
    source = trainer.read_text()
    revised = replace_once(source,
        """        if alpha == 1.0:
            local_scale = beta / math.sqrt(trainable_dimension)
""",
        """        if alpha == 1.0:
            if getattr(args, "zo_estimator", "legacy") == "isotropic_raw":
                # Paper Eq. (ZOO): z ~ N(0,I_d). This branch ignores cloud data.
                local_component = [beta * v for v in v_params]
                self._log_and_validate_perturbation_components(
                    v_params, local_component, None, local_component
                )
                return local_component
            local_scale = beta / math.sqrt(trainable_dimension)
""")
    revised = replace_once(revised,
        '"[ZGR] parameterization=normalized_z_h; trainable dimension n=%d; "\n'
        '            "beta=%.8g; estimator=central_fd_times_z_h; mode=%s; outer_multiplier=%.8g",\n'
        '            self.trainable_params_number,',
        '"[ZGR] parameterization=%s; trainable dimension n=%d; "\n'
        '            "beta=%.8g; estimator=central_fd_times_z_h; mode=%s; outer_multiplier=%.8g",\n'
        '            "raw_isotropic_z" if getattr(args, "zo_estimator", "legacy") == "isotropic_raw" else "normalized_z_h",\n'
        '            self.trainable_params_number,')

    original_main = main_file.read_text()
    revised_main = replace_once(original_main,
        """    dm = TextClassificationDataManager(args, model_args, preprocessor, process_id, args.client_num_per_round)
    train_data_num, train_data_global, test_data_global, train_data_local_num_dict, \\
    train_data_local_dict, test_data_local_dict, num_clients = dm.load_cloud_server_client_data(process_id=process_id)
""",
        """    if process_id == 0 and float(args.alpha) == 1.0:
        # Rank 0 remains an MPI barrier but must not read private examples.
        if args.cloud_data_file_path or args.cloud_partition_file_path:
            raise ValueError("no-guidance run must not provide cloud guidance data")
        logging.info("[ZGR] pure ZOO: cloud rank has no training data")
        train_data_num, train_data_global, test_data_global = 0, None, None
        train_data_local_num_dict = train_data_local_dict = test_data_local_dict = None
        num_clients = args.client_num_in_total
    else:
        dm = TextClassificationDataManager(args, model_args, preprocessor, process_id, args.client_num_per_round)
        train_data_num, train_data_global, test_data_global, train_data_local_num_dict, \\
        train_data_local_dict, test_data_local_dict, num_clients = dm.load_cloud_server_client_data(process_id=process_id)
""")
    before = {str(trainer.relative_to(repo)): digest_bytes(source.encode()),
              str(main_file.relative_to(repo)): digest_bytes(original_main.encode())}
    trainer.write_text(revised)
    main_file.write_text(revised_main)
    after = {str(trainer.relative_to(repo)): digest_bytes(revised.encode()),
             str(main_file.relative_to(repo)): digest_bytes(revised_main.encode())}
    print(json.dumps({"source_commit": head, "before_sha256": before,
                      "after_sha256": after,
                      "diff": subprocess.check_output(["git", "diff", "--stat"],
                                                      cwd=repo, text=True)}, indent=2))


if __name__ == "__main__":
    main()

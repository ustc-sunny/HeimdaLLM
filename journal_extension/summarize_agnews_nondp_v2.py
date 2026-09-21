#!/usr/bin/env python3
"""Summarize all predeclared AG News v2 seeds, requiring completed controls."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics

SEEDS = (57, 58, 59)
ARMS = ('no_cloud', 'client_syn', 'same_source_real_matched', 'real_matched', 'shuffled_label')

def read(path):
    return json.loads(path.read_text())

def summarize(results_root):
    records = []
    differences = []
    for seed in SEEDS:
        root = results_root / ('matpool_agnews_nondp_v2_seed%d' % seed)
        seed_dir = root / ('seed_%d' % seed)
        validation = read(seed_dir / 'cross_arm_validation.json')
        if validation['status'] != 'complete' or validation['errors']:
            raise ValueError('Cross-arm validation failed: %s' % root)
        if validation['cross_arm']['client_objective_queries_per_arm'] != 300:
            raise ValueError('Unexpected query budget')
        manifest = read(seed_dir / 'synthetic/generated/manifest.json')
        if (manifest['status'] != 'complete' or not manifest['is_true_non_dp']
                or manifest['synthetic_records'] != 128):
            raise ValueError('Unexpected generation status or budget')
        content = (seed_dir / 'synthetic/generated/synthetic.jsonl').read_bytes()
        if hashlib.sha256(content).hexdigest() != manifest['synthetic_sha256']:
            raise ValueError('Synthetic hash mismatch')
        final = {}
        for arm in ARMS:
            metrics = read(seed_dir / 'arms' / arm / 'metrics.json')
            if metrics['status'] != 'complete' or not all(metrics['checks'].values()):
                raise ValueError('Incomplete arm: %s seed %d' % (arm, seed))
            value = metrics['final']
            matrix = value['confusion_matrix']
            total = sum(sum(row) for row in matrix)
            correct = sum(matrix[i][i] for i in range(4))
            if total != 512 or abs(correct / total - value['acc']) > 1e-12:
                raise ValueError('Accuracy/confusion matrix mismatch')
            final[arm] = value
            records.append(dict(seed=seed, arm=arm, accuracy=value['acc'],
                                macro_f1=value['macro_f1'], loss=value['eval_loss']))
        differences.append(dict(
            seed=seed,
            synthetic_minus_no_cloud_pp=100 * (final['client_syn']['acc'] - final['no_cloud']['acc']),
            synthetic_minus_shuffled_pp=100 * (final['client_syn']['acc'] - final['shuffled_label']['acc'])))
    aggregate = {}
    for arm in ARMS:
        values = [r['accuracy'] * 100 for r in records if r['arm'] == arm]
        aggregate[arm] = dict(mean_accuracy_pct=statistics.mean(values),
                              sample_sd_accuracy_pp=statistics.stdev(values),
                              min_accuracy_pct=min(values), max_accuracy_pct=max(values))
    paired = {}
    for name in ('synthetic_minus_no_cloud_pp', 'synthetic_minus_shuffled_pp'):
        values = [d[name] for d in differences]
        paired[name] = dict(mean=statistics.mean(values), min=min(values), max=max(values),
                            positive_seed_count=sum(v > 0 for v in values))
    return dict(status='complete', seeds=list(SEEDS), records=records,
                paired_differences=differences, aggregate=aggregate, paired_summary=paired,
                scope='Three seeds on fixed source/development clients; five rounds; Non-DP. Not final convergence or statistical significance.')

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-root', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    result = summarize(args.results_root)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')
    lines = ['# AG News Non-DP v2 results', '', result['scope'], '',
             '| Seed | No guidance | Synthetic | Same-source real | Reserved real | Shuffled |',
             '|---|---:|---:|---:|---:|---:|']
    for seed in SEEDS:
        values = {r['arm']: r['accuracy'] * 100 for r in result['records'] if r['seed'] == seed}
        lines.append('| %d | ' % seed + ' | '.join('%.4f%%' % values[a] for a in ARMS) + ' |')
    lines += ['| Mean | ' + ' | '.join('%.4f%%' % result['aggregate'][a]['mean_accuracy_pct'] for a in ARMS) + ' |', '',
              '| Seed | Synthetic minus no guidance (pp) | Synthetic minus shuffled (pp) |',
              '|---|---:|---:|']
    for row in result['paired_differences']:
        lines.append('| {seed} | {synthetic_minus_no_cloud_pp:+.4f} | {synthetic_minus_shuffled_pp:+.4f} |'.format(**row))
    lines += ['', 'All 15 arms passed completion checks and cross-arm validation. Each arm used 300 client objective queries.',
              'Synthetic hashes and final accuracy/confusion-matrix consistency were verified.',
              'Refer to AGNEWS_NONDP_V2_PLAN.md and QUALITATIVE_AUDIT.md for configuration and sample-audit limitations.', '']
    (args.output_dir / 'METRICS.md').write_text('\n'.join(lines))
    print(json.dumps(result['paired_summary'], indent=2))

if __name__ == '__main__':
    main()

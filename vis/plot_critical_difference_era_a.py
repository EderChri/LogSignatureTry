"""
plot_critical_difference_era_a.py — Critical-difference diagram over every
capture24 -> Opportunity method config from the "Era A" sweep: all (view
combo x window x encoder) results that used channel_adapt='none' and share
pretrain seeds {4..9} (n=6 paired instances).

Excludes the newer channel-adapt-aware sweep (tukey128 lead-lag/red-pen,
norm_win64), which used seeds {0..4} with channel_adapt in {pca, drop} — a
different setup with no seed overlap with Era A (see plot_critical_difference.py
for that comparison instead).

Usage: python vis/plot_critical_difference_era_a.py
"""

import re
import sys
import numpy as np
from scipy import stats

sys.path.insert(0, 'vis')
import visualize_capture24 as vc24
from plot_critical_difference import nemenyi_q_alpha, plot_cd_diagram_many

TSV = 'out_finetune/_DA_Opportunity_256_00/final_test_metric_summary.tsv'
ERA_A_SEEDS = list(range(4, 10))

VIEW_LABEL = {'v2dx_v3xf': 'dx+xf', 'v2logsig_v3xf': 'logsig+xf',
              'v2dx_v3logsig': 'dx+logsig', 'v2logsig_nview': 'logsig(2v)'}
ENC_LABEL = {'transformer': 'mean-pool', 'transformer_bilinear': 'Xformer-bil',
             'mlp_logsig': 'MLP-LogSig', 'mlp_logsig_bilinear': 'MLP-LogSig-bil'}


def label(key):
    view_key, window, encoder, ca = key
    return f'{VIEW_LABEL[view_key]}/{window}/{ENC_LABEL[encoder]}'


def build_matrix():
    groups = {}
    with open(TSV) as f:
        for line in f:
            line = line.rstrip('\n')
            if not line or line.startswith('#'):
                continue
            parts = line.split('\t')
            if len(parts) < 2:
                continue
            parsed = vc24.parse_finetune_row(parts[0])
            if parsed is None or parsed[3] != 'none':
                continue
            m = re.search(r'_ep\d+_(\d+)_', parts[0])
            if not m:
                continue
            seed = int(m.group(1))
            if seed not in ERA_A_SEEDS:
                continue
            try:
                acc = float(parts[1])
            except ValueError:
                continue
            groups.setdefault(parsed, {}).setdefault(seed, []).append(acc)

    complete = {k: v for k, v in groups.items() if set(v.keys()) == set(ERA_A_SEEDS)}
    dropped = {k: sorted(v.keys()) for k, v in groups.items() if set(v.keys()) != set(ERA_A_SEEDS)}
    if dropped:
        print(f'Dropped {len(dropped)} incomplete-coverage group(s):')
        for k, seeds in dropped.items():
            print(f'  {label(k)}: seeds present {seeds}')

    keys = list(complete.keys())
    n = len(ERA_A_SEEDS)
    data = np.zeros((n, len(keys)))
    for j, key in enumerate(keys):
        for i, s in enumerate(ERA_A_SEEDS):
            vals = complete[key][s]
            data[i, j] = sum(vals) / len(vals)
    return [label(k) for k in keys], data


def main():
    methods, data = build_matrix()
    n, k = data.shape

    ranks = np.zeros_like(data)
    for i in range(n):
        ranks[i] = stats.rankdata(-data[i])
    avg_ranks = dict(zip(methods, ranks.mean(axis=0)))

    stat, p = stats.friedmanchisquare(*[data[:, j] for j in range(k)])
    cd = nemenyi_q_alpha(k) * np.sqrt(k * (k + 1) / (6.0 * n))

    print(f'k={k} methods, n={n} paired seeds')
    print('Average ranks (best to worst):')
    for m, r in sorted(avg_ranks.items(), key=lambda x: x[1]):
        print(f'  {r:6.3f}  {m}')
    print(f'Friedman chi2={stat:.4f}, p={p:.3g}')
    print(f'Nemenyi CD (alpha=0.05, k={k}, n={n}) = {cd:.4f}')

    stats_line = (f'Friedman χ²={stat:.2f}, p={p:.3g}  (k={k} methods, n={n} paired seeds)  '
                  'thick bars = groups not significantly different (Nemenyi, α=0.05)')
    plot_cd_diagram_many(
        avg_ranks, cd, k, n,
        title='Critical difference — capture24 -> Opportunity, Era A sweep (channel_adapt=none, seeds 4-9)',
        out_path='plots/critical_difference_opportunity_era_a.png',
        stats_line=stats_line,
    )


if __name__ == '__main__':
    main()

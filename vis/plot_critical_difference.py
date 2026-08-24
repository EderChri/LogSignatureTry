"""
plot_critical_difference.py — Critical-difference diagram (Demsar 2006 style):
average ranks of the dx+xf baseline vs. the tukey128 adjustment sweep
(none / reduced-pen / lead-lag / lead-lag+red-pen), on capture24 -> Opportunity.

Scope note: USC-HAD is excluded — its dx+xf baseline was pretrained on seeds
{5..9} while the tukey128 sweep used seeds {0..4}, so no seed-matched instance
exists there. Opportunity's baseline covers seeds {0..9}, overlapping the
sweep's {0..4}, giving 5 paired instances. Each instance's method value is the
mean accuracy across that method's hyperparameter reruns sharing the seed
(rp strength / lead-lag window), since those aren't repeated measures of the
same config.

Usage: python vis/plot_critical_difference.py
"""

import re
import sys
import numpy as np
from scipy import stats
import matplotlib
import matplotlib.pyplot as plt

sys.path.insert(0, 'vis')
import visualize_capture24 as vc24

METHOD_KEYS = {
    'dx+xf (baseline)':   ('v2dx_v3xf', 'global', 'transformer', 'none'),
    'tukey128':           ('v2logsig_nview', 'tukey128', 'mlp_logsig_bilinear', 'pca'),
    'tukey128+red.pen.':  ('v2logsig_nview', 'tukey128_rp', 'mlp_logsig_bilinear', 'pca'),
    'tukey128+lead-lag':  ('v2logsig_nview', 'tukey128_ll', 'mlp_logsig_bilinear', 'pca'),
    'tukey128+ll+rp':     ('v2logsig_nview', 'tukey128_ll_rp', 'mlp_logsig_bilinear', 'pca'),
}
SHARED_SEEDS = range(5)   # overlap between baseline and sweep seed sets
TSV = 'out_finetune/_DA_Opportunity_256_00/final_test_metric_summary.tsv'


def nemenyi_q_alpha(k: int, alpha: float = 0.05) -> float:
    """Studentized-range critical value / sqrt(2), for any k (matches the
    standard Nemenyi table for k<=10; extends via the exact distribution
    beyond it instead of being capped at a hardcoded table)."""
    return stats.studentized_range.ppf(1 - alpha, k, np.inf) / np.sqrt(2)


def extract_seed_scores(tsv_path, target_key):
    out = {}
    with open(tsv_path) as f:
        for line in f:
            line = line.rstrip('\n')
            if not line or line.startswith('#'):
                continue
            parts = line.split('\t')
            if len(parts) < 2:
                continue
            parsed = vc24.parse_finetune_row(parts[0])
            if parsed != target_key:
                continue
            m = re.search(r'_ep\d+_(\d+)_', parts[0])
            if not m:
                continue
            seed = int(m.group(1))
            try:
                acc = float(parts[1])
            except ValueError:
                continue
            out.setdefault(seed, []).append(acc)
    return out


def build_matrix():
    methods = list(METHOD_KEYS.keys())
    per_seed = {s: {} for s in SHARED_SEEDS}
    for name, key in METHOD_KEYS.items():
        seeds = extract_seed_scores(TSV, key)
        for s in SHARED_SEEDS:
            vals = seeds.get(s, [])
            if not vals:
                raise ValueError(f'missing data for {name} seed {s}')
            per_seed[s][name] = sum(vals) / len(vals)
    data = np.array([[per_seed[s][m] for m in methods] for s in SHARED_SEEDS])
    return methods, data


def maximal_cliques(sorted_methods, sorted_ranks, cd):
    """Return list of (start_idx, end_idx) index ranges into sorted_ranks
    such that ranks[end]-ranks[start] <= cd, keeping only maximal groups."""
    n = len(sorted_ranks)
    groups = []
    for i in range(n):
        j = i
        while j + 1 < n and sorted_ranks[j + 1] - sorted_ranks[i] <= cd:
            j += 1
        if j > i:
            groups.append((i, j))
    # drop groups fully contained in another
    maximal = []
    for g in groups:
        if not any(g != h and h[0] <= g[0] and g[1] <= h[1] for h in groups):
            maximal.append(g)
    # de-duplicate
    return sorted(set(maximal))


def plot_cd_diagram(avg_ranks: dict, cd: float, k: int, n: int, title: str, out_path: str,
                     friedman_stat: float, friedman_p: float):
    methods = list(avg_ranks.keys())
    ranks = np.array([avg_ranks[m] for m in methods])
    order = np.argsort(ranks)
    sorted_methods = [methods[i] for i in order]
    sorted_ranks = ranks[order]

    lo, hi = 1, k
    fig, ax = plt.subplots(figsize=(9, 3.6))

    INK = '#222222'
    MUTED = '#777777'
    LINE = '#333333'

    axis_y = 0.72
    ax.plot([lo, hi], [axis_y, axis_y], color=LINE, linewidth=1.4, zorder=2)
    for r in range(lo, hi + 1):
        ax.plot([r, r], [axis_y - 0.015, axis_y + 0.015], color=LINE, linewidth=1.2, zorder=2)
        ax.text(r, axis_y + 0.035, str(r), ha='center', va='bottom', fontsize=9, color=INK)

    # stems + labels, alternating up/down to reduce collisions
    label_slots_up = [axis_y + 0.16, axis_y + 0.30]
    label_slots_down = [axis_y - 0.16, axis_y - 0.30]
    up_i, down_i = 0, 0
    for idx, m in enumerate(sorted_methods):
        r = sorted_ranks[idx]
        go_up = idx % 2 == 0
        if go_up:
            y = label_slots_up[up_i % len(label_slots_up)]
            up_i += 1
            va = 'bottom'
        else:
            y = label_slots_down[down_i % len(label_slots_down)]
            down_i += 1
            va = 'top'
        ax.plot([r, r], [axis_y, y], color=MUTED, linewidth=0.9, zorder=1)
        ax.text(r, y, f'{m}\n(rank {r:.2f})', ha='center', va=va, fontsize=8.5, color=INK)

    # clique bars below the axis
    groups = maximal_cliques(sorted_methods, sorted_ranks, cd)
    bar_y0 = axis_y - 0.42
    bar_gap = 0.05
    for gi, (i, j) in enumerate(groups):
        y = bar_y0 - gi * bar_gap
        ax.plot([sorted_ranks[i], sorted_ranks[j]], [y, y], color=INK, linewidth=3.0,
                solid_capstyle='butt', zorder=3)

    # CD scale bar
    cd_y = bar_y0 - len(groups) * bar_gap - 0.12
    cd_x0 = lo
    ax.plot([cd_x0, cd_x0 + cd], [cd_y, cd_y], color=INK, linewidth=1.6, zorder=3)
    ax.plot([cd_x0, cd_x0], [cd_y - 0.02, cd_y + 0.02], color=INK, linewidth=1.6, zorder=3)
    ax.plot([cd_x0 + cd, cd_x0 + cd], [cd_y - 0.02, cd_y + 0.02], color=INK, linewidth=1.6, zorder=3)
    ax.text(cd_x0 + cd / 2, cd_y - 0.05, f'CD = {cd:.3f}', ha='center', va='top',
            fontsize=8.5, color=INK)

    ax.set_xlim(lo - 0.5, hi + 0.5)
    ax.set_ylim(cd_y - 0.18, axis_y + 0.42)
    ax.axis('off')
    ax.set_title(title, fontsize=11, fontweight='bold', pad=4)
    ax.text(0.5, -0.04,
            f'Friedman χ²={friedman_stat:.2f}, p={friedman_p:.4f}  (k={k} methods, n={n} paired seeds)  '
            'thick bars = groups not significantly different (Nemenyi, α=0.05)',
            transform=ax.transAxes, ha='center', va='top', fontsize=8, color=MUTED)

    plt.tight_layout()
    import os
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    base, _ = __import__('os.path', fromlist=['splitext']).splitext(out_path)
    if not out_path.endswith('.pdf'):
        plt.savefig(base + '.pdf', bbox_inches='tight')
    print(f'Saved -> {out_path}')


def plot_cd_diagram_many(avg_ranks: dict, cd, k: int, n: int, title: str, out_path: str,
                          stats_line: str, subtitle: str = '', groups=None):
    """Classic many-classifier CD diagram: methods sorted by rank, split into
    a left column (best half) and right column (worst half), each connected
    to its rank position on the axis via an elbow leader line. Clique bars
    (groups not significantly different) are drawn as thick segments above
    the axis.

    groups: precomputed list of (start_idx, end_idx) index pairs into the
    rank-sorted method list (as returned by maximal_cliques or a graph-based
    equivalent). If None, falls back to interval-based cliques from `cd`
    (Nemenyi). If `cd` is None, the CD scale-bar legend is omitted (used for
    the pairwise Wilcoxon-Holm view, which has no single distance measure)."""
    methods = list(avg_ranks.keys())
    ranks = np.array([avg_ranks[m] for m in methods])
    order = np.argsort(ranks)
    sorted_methods = [methods[i] for i in order]
    sorted_ranks = ranks[order]

    lo, hi = 1, k
    half = (k + 1) // 2
    left_idx = list(range(half))            # best half
    right_idx = list(range(half, k))        # worst half
    n_rows = max(len(left_idx), len(right_idx), 1)

    INK = '#222222'
    MUTED = '#777777'
    LINE = '#333333'

    # Single coordinate frame: axis at y=0. Rows of labels descend below it
    # (y<0); significance bars + CD legend sit above it (y>0). row_h is fixed
    # in absolute units so figure height (set from n_rows below) always
    # matches the space the rows actually need.
    row_h = 0.26
    axis_y = 0.0
    label_x_margin = max(4.0, k * 0.14)
    label_x_left = lo - label_x_margin
    label_x_right = hi + label_x_margin

    if groups is None:
        groups = maximal_cliques(sorted_methods, sorted_ranks, cd)
    bar_gap = 0.16
    above_axis_top = 0.35 + len(groups) * bar_gap + 0.35   # ticks/CD-bar/clique-bars headroom
    below_axis_bottom = -(0.30 + (n_rows - 1) * row_h + 0.15)

    fig_h = (above_axis_top - below_axis_bottom) * 0.62 + 1.0
    fig, ax = plt.subplots(figsize=(13, fig_h))

    ax.plot([lo, hi], [axis_y, axis_y], color=LINE, linewidth=1.4, zorder=2)
    tick_step = 1 if k <= 15 else 5
    for r in range(lo, hi + 1):
        tick_len = 0.02 if r % tick_step == 0 else 0.01
        ax.plot([r, r], [axis_y - tick_len, axis_y + tick_len], color=LINE, linewidth=1.0, zorder=2)
        if r % tick_step == 0 or r == hi:
            ax.text(r, axis_y + 0.05, str(r), ha='center', va='bottom', fontsize=7.5, color=INK)

    def draw_side(idx_list, x_label, ha):
        for row, i in enumerate(idx_list):
            r = sorted_ranks[i]
            y = -(0.30 + row * row_h)
            ax.plot([r, r], [axis_y, y], color=MUTED, linewidth=0.8, zorder=1)
            ax.plot([r, x_label], [y, y], color=MUTED, linewidth=0.8, zorder=1)
            ax.text(x_label + (0.3 if ha == 'left' else -0.3), y,
                     f'{sorted_methods[i]}  ({r:.2f})',
                     ha=ha, va='center', fontsize=7.3, color=INK)

    draw_side(left_idx, label_x_left, 'right')
    draw_side(right_idx, label_x_right, 'left')

    # clique bars, stacked above the axis
    bar_y0 = 0.20
    for gi, (i, j) in enumerate(groups):
        y = bar_y0 + gi * bar_gap
        ax.plot([sorted_ranks[i], sorted_ranks[j]], [y, y], color=INK, linewidth=2.6,
                solid_capstyle='butt', zorder=3)

    # CD scale bar, above the clique bars, right-aligned over the axis (Nemenyi only)
    if cd is not None:
        cd_y = bar_y0 + len(groups) * bar_gap + 0.15
        cd_x0 = hi - cd
        ax.plot([cd_x0, cd_x0 + cd], [cd_y, cd_y], color=INK, linewidth=1.4, zorder=3)
        ax.plot([cd_x0, cd_x0], [cd_y - 0.02, cd_y + 0.02], color=INK, linewidth=1.4, zorder=3)
        ax.plot([cd_x0 + cd, cd_x0 + cd], [cd_y - 0.02, cd_y + 0.02], color=INK, linewidth=1.4, zorder=3)
        ax.text(cd_x0 + cd / 2, cd_y + 0.03, f'CD = {cd:.2f}', ha='center', va='bottom', fontsize=8, color=INK)

    ax.set_xlim(label_x_left - 0.5, label_x_right + 0.5)
    ax.set_ylim(below_axis_bottom, above_axis_top)
    ax.axis('off')
    ax.set_title(title, fontsize=11, fontweight='bold', pad=4)
    fig.text(0.5, 0.005, stats_line + ('\n' + subtitle if subtitle else ''),
             ha='center', va='bottom', fontsize=8, color=MUTED)

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    import os
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    base, _ = __import__('os.path', fromlist=['splitext']).splitext(out_path)
    if not out_path.endswith('.pdf'):
        plt.savefig(base + '.pdf', bbox_inches='tight')
    print(f'Saved -> {out_path}')


def main():
    methods, data = build_matrix()
    n, k = data.shape

    ranks = np.zeros_like(data)
    for i in range(n):
        ranks[i] = stats.rankdata(-data[i])
    avg_ranks = dict(zip(methods, ranks.mean(axis=0)))

    stat, p = stats.friedmanchisquare(*[data[:, j] for j in range(k)])
    cd = nemenyi_q_alpha(k) * np.sqrt(k * (k + 1) / (6.0 * n))

    print('Average ranks:')
    for m, r in sorted(avg_ranks.items(), key=lambda x: x[1]):
        print(f'  {m:22s} {r:.3f}')
    print(f'Friedman chi2={stat:.4f}, p={p:.4f}')
    print(f'Nemenyi CD (alpha=0.05) = {cd:.4f}')

    plot_cd_diagram(
        avg_ranks, cd, k, n,
        title='Critical difference — capture24 -> Opportunity (tukey128 adjustment sweep)',
        out_path='plots/critical_difference_opportunity.png',
        friedman_stat=stat, friedman_p=p,
    )


if __name__ == '__main__':
    main()

"""
visualize_capture24.py — Test accuracy for capture24 pretrain → all finetune targets.

Usage:
  python visualize_capture24.py                 # all available datasets
  python visualize_capture24.py --datasets WISDM HAR70plus
  python visualize_capture24.py --metric f1_score

Layout: one subplot per dataset, same bar structure as visualize_results.py.
Missing datasets are skipped without error.
"""

import argparse
import os
import re
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats

matplotlib.rcParams['hatch.linewidth'] = 2.0

# ---------------------------------------------------------------------------
# Datasets (in display order)
# ---------------------------------------------------------------------------

ALL_DATASETS = [
    ('WISDM',       '_DA_WISDM_256_00',       'WISDM'),
    ('WISDM2',      '_DA_WISDM2_256_00',       'WISDM2'),
    ('HAR70plus',   '_DA_HAR70plus_256_00',    'HAR70plus'),
    ('USC_HAD',     '_DA_USC_HAD_256_00',      'USC-HAD'),
    ('Opportunity', '_DA_Opportunity_256_00',  'Opportunity\n(113ch→32 PCA)'),
    ('Skoda',       '_DA_Skoda_256_00',        'Skoda\n(60ch→4 PCA)'),
]

PRETRAIN_TAG = '_DA_capture24_256_00'   # full capture24 only (not mini)

# ---------------------------------------------------------------------------
# View / window / encoder constants (shared with visualize_results.py)
# ---------------------------------------------------------------------------

VIEW_ORDER = ['v2dx_v3xf', 'v2logsig_v3xf', 'v2dx_v3logsig', 'v2logsig_nview']
VIEW_LABEL = {
    'v2dx_v3xf':      'dx+xf\n(baseline)',
    'v2logsig_v3xf':  'logsig+xf',
    'v2dx_v3logsig':  'dx+logsig',
    'v2logsig_nview': 'logsig\n(2-view)',
}

WINDOW_ORDER = ['global', 'win64', 'win128', 'win128_s7', 'tukey64', 'tukey128', 'tukey128_s7']
WINDOW_LABEL = {
    'global':      'global',
    'win64':       'win 64',
    'win128':      'win 128',
    'win128_s7':   'win 128\ns7',
    'tukey64':     'tukey 64',
    'tukey128':    'tukey 128',
    'tukey128_s7': 'tukey 128\ns7',
}
WINDOW_COLOR = {
    'global':      '#4C72B0',
    'win64':       '#DD8452',
    'win128':      '#55A868',
    'win128_s7':   '#2CA02C',
    'tukey64':     '#C44E52',
    'tukey128':    '#8172B3',
    'tukey128_s7': '#6B4C9A',
}

ENCODERS = ['transformer', 'transformer_plast', 'mlp_logsig',
            'mlp_logsig_bilinear', 'transformer_bilinear']
ENC_HATCH = {
    'transformer':          '',
    'transformer_plast':    '\\' * 6,
    'mlp_logsig':           '/' * 6,
    'mlp_logsig_bilinear':  'xx',
    'transformer_bilinear': 'oo',
}
ENC_LABEL = {
    'transformer':          'Transformer (mean pool)',
    'transformer_plast':    'Transformer (last pool)',
    'mlp_logsig':           'MLP-LogSig',
    'mlp_logsig_bilinear':  'MLP-LogSig bilinear',
    'transformer_bilinear': 'Transformer bilinear',
}

# ---------------------------------------------------------------------------
# Bar geometry (mirrors visualize_results.py)
# ---------------------------------------------------------------------------

BAR_W     = 0.048
INNER_GAP = 0.008
WIN_GAP   = 0.025
VIEW_SEP  = 1.20

SUB_W_2 = 2 * BAR_W + INNER_GAP
SUB_W_3 = 3 * BAR_W + 2 * INNER_GAP
_BIL_WIN_SET = {'win64', 'win128', 'tukey128'}
LOGSIG_SPAN = (
    SUB_W_3
    + sum(SUB_W_3 if w in _BIL_WIN_SET else SUB_W_2 for w in WINDOW_ORDER[1:])
    + (len(WINDOW_ORDER) - 1) * WIN_GAP
)


def _view_bar_specs(view_key):
    if view_key == 'v2dx_v3xf':
        # Single transformer (attention, mean-pool) bar — the true baseline.
        # transformer_bilinear is a second bar if data exists (for comparison).
        hw = BAR_W / 2 + INNER_GAP / 2
        return [
            ('global', 'transformer',          -hw),
            ('global', 'transformer_bilinear',  +hw),
        ]
    specs = []
    x = -LOGSIG_SPAN / 2
    for i, win in enumerate(WINDOW_ORDER):
        if i > 0:
            x += WIN_GAP
        is_bil = win in _BIL_WIN_SET
        sw = SUB_W_3 if (win == 'global' or is_bil) else SUB_W_2
        xc = x + sw / 2
        if win == 'global':
            specs.append((win, 'transformer',       xc - BAR_W - INNER_GAP))
            specs.append((win, 'transformer_plast', xc))
            specs.append((win, 'mlp_logsig',        xc + BAR_W + INNER_GAP))
        elif is_bil:
            specs.append((win, 'transformer',          xc - BAR_W - INNER_GAP))
            specs.append((win, 'mlp_logsig',           xc))
            specs.append((win, 'mlp_logsig_bilinear',  xc + BAR_W + INNER_GAP))
        else:
            specs.append((win, 'transformer', xc - BAR_W / 2 - INNER_GAP / 2))
            specs.append((win, 'mlp_logsig',  xc + BAR_W / 2 + INNER_GAP / 2))
        x += sw
    return specs


_BAR_SPECS = {v: _view_bar_specs(v) for v in VIEW_ORDER}

# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def parse_finetune_row(name: str):
    """Return (view_key, window, encoder) or None. Only capture24_256_00 pretrain."""
    if not name or not name.endswith('_finetune'):
        return None
    if PRETRAIN_TAG not in name:
        return None
    if any(m in name for m in ('_ilcrosstime_', '_ilviewembed_')):
        return None

    is_bilinear = '_ilbilinear_' in name
    if '_mlp_logsig_' in name:
        encoder = 'mlp_logsig_bilinear' if is_bilinear else 'mlp_logsig'
    elif '_plast' in name:
        encoder = 'transformer_plast'
    elif is_bilinear:
        encoder = 'transformer_bilinear'
    else:
        encoder = 'transformer'

    m = re.search(r'(v2[a-z]+_(?:v3[a-z]+|nview))_ep', name)
    if not m or m.group(1) not in VIEW_LABEL:
        return None
    view_key = m.group(1)

    window = 'global'
    for w in ('win128_s7', 'tukey128_s7', 'win64', 'win128', 'tukey64', 'tukey128'):
        if f'_{w}_' in name or f'_{w}.' in name:
            window = w
            break

    return view_key, window, encoder


def read_tsv(path: str):
    """Return dict (view_key, window, encoder) → list[float]."""
    scores = {}
    try:
        with open(path) as f:
            for line in f:
                line = line.rstrip('\n')
                if not line or line.startswith('#') or line.startswith('run_name'):
                    continue
                parts = line.split('\t')
                if len(parts) < 2:
                    continue
                parsed = parse_finetune_row(parts[0])
                if parsed is None:
                    continue
                try:
                    score = float(parts[1])
                except ValueError:
                    continue
                scores.setdefault(parsed, []).append(score)
    except FileNotFoundError:
        pass
    return scores


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _mean_ci(vals):
    """Return (mean, half_ci_95) or (None, None)."""
    if not vals:
        return None, None
    arr = np.array(vals, dtype=float)
    mu = arr.mean()
    if len(arr) >= 2:
        se = stats.sem(arr)
        ci = se * stats.t.ppf(0.975, len(arr) - 1)
    else:
        ci = 0.0
    return mu, ci


def plot_dataset(ax, scores, title, metric_label, baseline_mean=None):
    """Draw one bar-chart panel."""
    view_centers = {v: i * VIEW_SEP for i, v in enumerate(VIEW_ORDER)}

    for view_key in VIEW_ORDER:
        cx = view_centers[view_key]
        for (win, enc, rel_x) in _BAR_SPECS[view_key]:
            vals = scores.get((view_key, win, enc), [])
            mu, ci = _mean_ci(vals)
            if mu is None:
                continue
            color = WINDOW_COLOR[win]
            hatch = ENC_HATCH[enc]
            ax.bar(cx + rel_x, mu, BAR_W, color=color, hatch=hatch,
                   edgecolor='white', linewidth=0.4, alpha=0.9)
            ax.errorbar(cx + rel_x, mu, yerr=ci, fmt='none',
                        ecolor='black', elinewidth=0.8, capsize=2)

    # Baseline reference line
    if baseline_mean is not None:
        ax.axhline(baseline_mean, color='black', linestyle=':', linewidth=1.2,
                   label=f'dx+xf baseline ({baseline_mean:.3f})')

    # x-axis labels
    ax.set_xticks([view_centers[v] for v in VIEW_ORDER])
    ax.set_xticklabels([VIEW_LABEL[v] for v in VIEW_ORDER], fontsize=8)
    ax.set_xlim(-VIEW_SEP * 0.6, (len(VIEW_ORDER) - 1) * VIEW_SEP + VIEW_SEP * 0.6)

    ax.set_ylabel(metric_label, fontsize=9)
    ax.set_title(title, fontsize=10, fontweight='bold')
    ax.set_ylim(0.7, 1.05)
    ax.yaxis.grid(True, linewidth=0.4, alpha=0.5)
    ax.set_axisbelow(True)
    ax.spines[['top', 'right']].set_visible(False)

    # Seed count annotation on first bar group
    cx0 = view_centers[VIEW_ORDER[0]]
    all_n = [len(scores.get((VIEW_ORDER[0], 'global', e), []))
             for e in ['transformer', 'transformer_bilinear']]
    n = max(all_n) if all_n else 0
    if n > 0:
        ax.text(cx0, 0.02, f'n={n}', ha='center', va='bottom',
                fontsize=7, color='gray', transform=ax.get_xaxis_transform())


def build_legend(fig):
    """Shared legend for window colors and encoder hatches."""
    win_patches = [mpatches.Patch(color=WINDOW_COLOR[w], label=WINDOW_LABEL[w])
                   for w in WINDOW_ORDER]
    enc_patches = [mpatches.Patch(facecolor='lightgray', hatch=ENC_HATCH[e],
                                  edgecolor='black', label=ENC_LABEL[e])
                   for e in ENCODERS if e in ENC_HATCH]
    handles = win_patches + [mpatches.Patch(fill=False, label='')] + enc_patches
    fig.legend(handles=handles, loc='lower center', ncol=6,
               fontsize=8, frameon=False, bbox_to_anchor=(0.5, -0.02))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets', nargs='+',
                        default=[d[0] for d in ALL_DATASETS],
                        choices=[d[0] for d in ALL_DATASETS],
                        help='Which datasets to plot (default: all available)')
    parser.add_argument('--metric', default='accuracy',
                        choices=['accuracy', 'f1_score'],
                        help='Metric column to visualise')
    parser.add_argument('--out', default='capture24_results.pdf',
                        help='Output file (pdf/png/svg)')
    args = parser.parse_args()

    # Load data for each requested dataset, skip if no TSV
    requested = {d[0] for d in ALL_DATASETS if d[0] in args.datasets}
    panels = []
    for (short, data_tag, label) in ALL_DATASETS:
        if short not in requested:
            continue
        tsv = f'out_finetune/{data_tag}/final_test_metric_summary.tsv'
        scores = read_tsv(tsv)
        if not scores:
            print(f'  Skipping {short}: no data in {tsv}')
            continue
        panels.append((label, scores))

    if not panels:
        print('No data found for any requested dataset.')
        return

    metric_label = 'Test accuracy' if args.metric == 'accuracy' else 'Test F1'

    ncols = min(3, len(panels))
    nrows = (len(panels) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(7 * ncols, 4.5 * nrows + 1.2),
                             squeeze=False)

    for idx, (label, scores) in enumerate(panels):
        ax = axes[idx // ncols][idx % ncols]
        # Prefer attention baseline; fall back to bilinear if not yet run
        baseline_vals = scores.get(('v2dx_v3xf', 'global', 'transformer'), [])
        if not baseline_vals:
            baseline_vals = scores.get(('v2dx_v3xf', 'global', 'transformer_bilinear'), [])
        baseline_mean, _ = _mean_ci(baseline_vals)
        plot_dataset(ax, scores, f'capture24 → {label}', metric_label, baseline_mean)

    # Hide unused subplots
    for idx in range(len(panels), nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    build_legend(fig)
    fig.suptitle(f'Multi-View Contrastive Learning — capture24 pretrain ({metric_label})',
                 fontsize=12, fontweight='bold', y=1.01)
    plt.tight_layout(rect=[0, 0.06, 1, 1])

    plt.savefig(args.out, bbox_inches='tight', dpi=150)
    print(f'Saved → {args.out}')
    print(f'Datasets plotted: {[p[0] for p in panels]}')


if __name__ == '__main__':
    main()

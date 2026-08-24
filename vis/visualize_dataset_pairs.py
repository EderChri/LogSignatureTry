"""
visualize_dataset_pairs.py — dx+xf (baseline) vs. logsig+xf, one row per
pretrain -> finetune dataset pair, with the random-init baseline for each
configuration shown in light grey behind the finetuned point.

Unlike visualize_capture24.py this is scoped to dataset pairs with matching
channel counts (no channel_adapt), only the two view combos that matter for
the dx-vs-logsig comparison (not dx+logsig or the 2-view logsig-only combo),
and only configurations backed by enough seeds to trust:

  - a configuration is shown only if it has >= MIN_SEEDS finetune seeds AND
    at least one matching random-init baseline run in the same TSV. Anything
    short of that (single-seed hyperparameter probes, orphaned finetune runs
    with no baseline counterpart, channel-adapt trials) is dropped rather
    than plotted half-supported.
  - the space freed up by dropping channel_adapt overlays, the dx+logsig /
    logsig-nview view groups, and per-encoder tick sub-structure goes into
    wider spacing between the points that remain.

Which metric to read is configurable per dataset pair (PAIR_METRIC below, or
--metric-override on the CLI); it defaults to accuracy. The aggregate TSVs
only ever store accuracy (run_finetune.py hardcodes monitoring_metric =
'accuracy'), so any other metric is recovered from the per-run result pickle
in out_finetune/{dataset}/{run_name} instead (metric_list[-1][2][name], the
final epoch's test-set metrics).

Usage:
  python vis/visualize_dataset_pairs.py
  python vis/visualize_dataset_pairs.py --pairs sleepeeg_epilepsy harth_har70plus
  python vis/visualize_dataset_pairs.py --metric f1_macro
  python vis/visualize_dataset_pairs.py --metric-override sleepeeg_epilepsy=auroc ecg_emg=f1_macro
  python vis/visualize_dataset_pairs.py --min-seeds 3
"""

import argparse
import os
import pickle
import re
from collections import defaultdict

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from scipy import stats

# ---------------------------------------------------------------------------
# Dataset pairs (in display order). Tags match datasets.cfg / preprocessed_data/.
# ---------------------------------------------------------------------------

_FDX = ['A', 'B', 'C', 'D']
_FDX_TAG = {c: f'_DA_FD-{c}-256_256_00' for c in _FDX}

DATASET_PAIRS = []
for _src in _FDX:
    for _dst in _FDX:
        if _src == _dst:
            continue
        DATASET_PAIRS.append((
            f'fd{_src.lower()}_fd{_dst.lower()}',
            _FDX_TAG[_src], _FDX_TAG[_dst],
            f'FD-{_src} → FD-{_dst}',
        ))

DATASET_PAIRS += [
    ('harth_har70plus', '_DA_HARTH_256_00', '_DA_HAR70plus_256_00', 'HARTH → HAR70plus'),
    ('har70plus_harth', '_DA_HAR70plus_256_00', '_DA_HARTH_256_00', 'HAR70plus → HARTH'),

    ('sleepeeg_epilepsy', '_DA_SleepEEG_256_00', '_DA_Epilepsy_256_00', 'SleepEEG → Epilepsy'),
    ('sleepeeg_emg',      '_DA_SleepEEG_256_00', '_DA_EMG_256_00',      'SleepEEG → EMG'),
    ('sleepeeg_fda',      '_DA_SleepEEG_256_00', _FDX_TAG['A'],         'SleepEEG → FD-A'),
    ('sleepeeg_fdb',      '_DA_SleepEEG_256_00', _FDX_TAG['B'],         'SleepEEG → FD-B'),
    ('sleepeeg_fdc',      '_DA_SleepEEG_256_00', _FDX_TAG['C'],         'SleepEEG → FD-C'),
    ('sleepeeg_fdd',      '_DA_SleepEEG_256_00', _FDX_TAG['D'],         'SleepEEG → FD-D'),

    ('ecg_epilepsy', '_DA_ECG_256_00', '_DA_Epilepsy_256_00', 'ECG → Epilepsy'),
    ('ecg_emg',      '_DA_ECG_256_00', '_DA_EMG_256_00',      'ECG → EMG'),
    ('ecg_fda',      '_DA_ECG_256_00', _FDX_TAG['A'],         'ECG → FD-A'),
    ('ecg_fdb',      '_DA_ECG_256_00', _FDX_TAG['B'],         'ECG → FD-B'),
    ('ecg_fdc',      '_DA_ECG_256_00', _FDX_TAG['C'],         'ECG → FD-C'),
    ('ecg_fdd',      '_DA_ECG_256_00', _FDX_TAG['D'],         'ECG → FD-D'),
]
PAIR_BY_KEY = {p[0]: p for p in DATASET_PAIRS}

MIN_SEEDS = 5

# ---------------------------------------------------------------------------
# Metric selection, per dataset pair. Edit here for a persistent default, or
# override per-run with --metric / --metric-override.
# ---------------------------------------------------------------------------

DEFAULT_METRIC = 'accuracy'
PAIR_METRIC = {
    # 'sleepeeg_epilepsy': 'f1_macro',
    'harth_har70plus': 'f1_subject_macro',
    'har70plus_harth': 'f1_subject_macro',
}

METRIC_LABEL = {
    'accuracy':         'Test accuracy',
    'f1_macro':         'Test F1 (macro)',
    'f1_weighted':      'Test F1 (weighted)',
    'precision':        'Test precision (macro)',
    'recall':           'Test recall (macro)',
    'auroc':             'Test AUROC',
    'auprc':             'Test AUPRC',
    'f1_subject_macro': 'Test F1 (per-subject mean)',
}
# Per-run result pickles changed their metric-dict key names over time
# (older runs wrote 'f1_score', current src/evaluation.py writes 'f1_macro').
METRIC_ALIASES = {
    'f1_macro': ['f1_macro', 'f1_score'],
}

VIEW_ORDER = ['v2dx_v3xf', 'v2logsig_v3xf']
VIEW_LABEL = {'v2dx_v3xf': 'dx + xf (baseline)', 'v2logsig_v3xf': 'logsig + xf'}

# ---------------------------------------------------------------------------
# Run-name parsing
# ---------------------------------------------------------------------------

RUN_RE = re.compile(
    r'^_DA_[\w-]+_\d+_\d+_pt-(?P<pt>_DA_[\w-]+_\d+_\d+)_'
    r'(?P<view>v2dx_v3xf|v2logsig_v3xf)_ep(?P<ep>\d+)_(?P<seed>\d+)_'
    r'(?:(?P<config>.+)_)?'
    r'(?:hidden|latent)_[A-Z]+_[\d.]+_\d+_'
    r'(?P<suffix>finetune|baseline|freeze)$'
)

# Earlier compute-environment migrations left full snapshots of run results
# behind (e.g. before moving to a new cluster) instead of being merged into
# out_finetune/. Some of that history — particularly dx+xf random-init
# baselines — was never rerun since, so it only exists in these archives.
# Search them all, preferring out_finetune/ when a run_name appears in more
# than one place (same experiment, not an independent extra seed).
RESULT_ROOTS = ['out_finetune', 'out_finetune_old', 'out_finetune_pre_canada_old']


def _candidate_dirs(finetune_tag: str):
    dirs = []
    for root in RESULT_ROOTS:
        for suffix in ('', '_old'):
            d = f'{root}/{finetune_tag}{suffix}'
            if os.path.exists(f'{d}/final_test_metric_summary.tsv'):
                dirs.append(d)
    return dirs


def load_pair(pretrain_tag: str, finetune_tag: str):
    """Return dict[(view, ep, config)] -> {'finetune': [(run_name, acc, src_dir)],
    'baseline': [...]}, restricted to configurations with >= MIN_SEEDS finetune
    seeds and at least one baseline run, merged across all result archives that
    have a final_test_metric_summary.tsv for this dataset. Returns None if no
    archive has one at all."""
    dirs = _candidate_dirs(finetune_tag)
    if not dirs:
        return None

    seen_runs = {}  # run_name -> (acc, src_dir), first-seen (= highest-priority) wins
    for d in dirs:
        with open(f'{d}/final_test_metric_summary.tsv') as f:
            for line in f:
                line = line.rstrip('\n')
                if not line or line.startswith('run_name'):
                    continue
                parts = line.split('\t')
                if len(parts) < 2:
                    continue
                run_name = parts[0]
                if run_name in seen_runs:
                    continue
                try:
                    acc = float(parts[1])
                except ValueError:
                    continue
                seen_runs[run_name] = (acc, d)

    buckets = defaultdict(lambda: defaultdict(list))
    for run_name, (acc, d) in seen_runs.items():
        m = RUN_RE.match(run_name)
        if not m or m.group('pt') != pretrain_tag:
            continue
        key = (m.group('view'), m.group('ep'), m.group('config') or '')
        buckets[key][m.group('suffix')].append((run_name, acc, d))

    result = {}
    for key, suffdict in buckets.items():
        ft = suffdict.get('finetune', [])
        bl = suffdict.get('baseline', [])
        if len(ft) >= MIN_SEEDS and len(bl) >= 1:
            result[key] = {'finetune': ft, 'baseline': bl}
    return result


# ---------------------------------------------------------------------------
# Metric value resolution (accuracy comes straight from the TSV; anything
# else is recovered from the per-run result pickle, read from whichever
# archive directory that run_name's score came from).
# ---------------------------------------------------------------------------

_pickle_metric_cache = {}


def _pickle_test_metrics(src_dir: str, run_name: str):
    cache_key = (src_dir, run_name)
    if cache_key in _pickle_metric_cache:
        return _pickle_metric_cache[cache_key]
    path = f'{src_dir}/{run_name}'
    te_m = None
    try:
        with open(path, 'rb') as f:
            _, _, metric_list = pickle.load(f)
        te_m = metric_list[-1][2]
    except Exception:
        pass
    _pickle_metric_cache[cache_key] = te_m
    return te_m


def metric_value(src_dir: str, run_name: str, acc_fallback: float, metric: str):
    if metric == 'accuracy':
        return acc_fallback
    te_m = _pickle_test_metrics(src_dir, run_name)
    if te_m is None:
        return None
    for alias in METRIC_ALIASES.get(metric, [metric]):
        val = te_m.get(alias)
        if val is not None:
            return float(val)
    return None


def summarize(entries, metric: str):
    vals = []
    for run_name, acc, src_dir in entries:
        v = metric_value(src_dir, run_name, acc, metric)
        if v is not None:
            vals.append(v)
    return vals


def mean_ci(vals):
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


# ---------------------------------------------------------------------------
# Human-readable labels for the literal config string (no semantic bucketing
# — every distinct config string is its own point, so hyperparameter
# variants never silently get merged together).
# ---------------------------------------------------------------------------

_TOKEN_LABEL = {
    'win64': 'win 64', 'win128': 'win 128',
    'tukey64': 'tukey 64', 'tukey128': 'tukey 128',
    'norm': 'norm', 'sl1': 'no lv1', 's7': 'stride 7',
}
_PARAM_LABEL = {'lag': 'lag', 'll': 'lead-lag', 'rp': 'red.pen.',
                'sp': 'taper', 'msp': 'min-taper', 'd': 'depth', 'cv': 'cv', 'lsn': 'noise'}
_PARAM_RE = re.compile(r'^(lag|ll|rp|sp|msp|d|cv|lsn)(\d+\.?\d*)$')

def window_core(config: str) -> str:
    toks = [t for t in config.split('_') if t and t not in ('mlp', 'logsig', 'ilbilinear')]
    return '_'.join(toks)


def humanize_window(core: str) -> str:
    if not core:
        return 'global'
    parts = []
    for tok in core.split('_'):
        if tok in _TOKEN_LABEL:
            parts.append(_TOKEN_LABEL[tok])
            continue
        m = _PARAM_RE.match(tok)
        if m:
            key, val = m.groups()
            parts.append(f'{_PARAM_LABEL.get(key, key)} {val}')
        else:
            parts.append(tok)
    return ', '.join(parts)


def humanize_encoder(config: str) -> str:
    if not config:
        return 'Transformer'
    is_mlp = 'mlp_logsig' in config
    bilinear = 'ilbilinear' in config
    base = 'MLP-LogSig' if is_mlp else 'Transformer'
    return base + (' +bilin.' if bilinear else '')


def point_label(key, show_ep: bool) -> str:
    _, ep, config = key
    lines = [humanize_encoder(config)]
    if show_ep:
        lines.append(f'pt ep{ep}')
    return '\n'.join(lines)


def sort_key(key):
    _, ep, config = key
    return (window_core(config), humanize_encoder(config), ep)


# ---------------------------------------------------------------------------
# Layout: two view groups (dx+xf, logsig+xf), one x-position per qualifying
# (view, ep, config) key found across the panels actually being plotted.
# ---------------------------------------------------------------------------

POINT_PITCH = 1.0
GROUP_GAP = 1.8
BRACKET_PAD = 0.45
EMPTY_GROUP_WIDTH = 0.8


def layout_group(keys):
    n = len(keys)
    if n == 0:
        return [], EMPTY_GROUP_WIDTH
    width = (n - 1) * POINT_PITCH
    return list(zip(keys, (i * POINT_PITCH - width / 2 for i in range(n)))), max(width, 0.6)


def build_layout(all_data):
    dx_keys, ls_keys = set(), set()
    for data in all_data:
        for k in data:
            (dx_keys if k[0] == 'v2dx_v3xf' else ls_keys).add(k)
    dx_sorted = sorted(dx_keys, key=sort_key)
    ls_sorted = sorted(ls_keys, key=sort_key)

    dx_items, dx_width = layout_group(dx_sorted)
    ls_items, ls_width = layout_group(ls_sorted)
    dx_center = 0.0
    ls_center = dx_center + dx_width / 2 + GROUP_GAP + ls_width / 2

    all_items = [(k, dx_center + rel) for k, rel in dx_items] + \
                [(k, ls_center + rel) for k, rel in ls_items]
    group_ranges = [
        ('v2dx_v3xf', dx_center, dx_width / 2 + BRACKET_PAD),
        ('v2logsig_v3xf', ls_center, ls_width / 2 + BRACKET_PAD),
    ]
    show_ep = len({k[1] for k in dx_keys | ls_keys}) > 1

    cores = sorted({window_core(k[2]) for k in dx_keys | ls_keys})
    palette = matplotlib.colormaps['tab10'].colors
    core_colors = {c: palette[i % len(palette)] for i, c in enumerate(cores)}

    return all_items, group_ranges, show_ep, core_colors


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

BASELINE_FACE = '#d9d9d9'
BASELINE_EDGE = '#999999'
CONNECTOR_COLOR = '#bbbbbb'
MARKER_SIZE = 7.5
MARKER = 'o'


def plot_pair(ax, label, data, metric, all_items, group_ranges, show_ep, core_colors):
    y_lo, y_hi = 1.0, 0.0

    for key, x in all_items:
        entry = data.get(key)
        if entry is None:
            continue
        ft_vals = summarize(entry['finetune'], metric)
        bl_vals = summarize(entry['baseline'], metric)
        if not ft_vals or not bl_vals:
            continue
        ft_mu, ft_ci = mean_ci(ft_vals)
        bl_mu, bl_ci = mean_ci(bl_vals)
        color = core_colors[window_core(key[2])]

        ax.plot([x, x], [bl_mu, ft_mu], color=CONNECTOR_COLOR, linewidth=1.2, zorder=2)

        bl_yerr = None if bl_ci is None else [[min(bl_ci, bl_mu)], [min(bl_ci, 1.0 - bl_mu)]]
        ax.errorbar(x, bl_mu, yerr=bl_yerr, fmt=MARKER, markersize=MARKER_SIZE,
                    markerfacecolor=BASELINE_FACE, markeredgecolor=BASELINE_EDGE,
                    markeredgewidth=1.0, ecolor=BASELINE_EDGE, elinewidth=1.0,
                    capsize=2, zorder=3, alpha=0.9)

        ft_yerr = None if ft_ci is None else [[min(ft_ci, ft_mu)], [min(ft_ci, 1.0 - ft_mu)]]
        ax.errorbar(x, ft_mu, yerr=ft_yerr, fmt=MARKER, markersize=MARKER_SIZE,
                    markerfacecolor=color, markeredgecolor='#333',
                    markeredgewidth=0.8, ecolor='#333', elinewidth=1.3,
                    capsize=2.5, zorder=5)

        ax.text(x, 0.012, f'n={len(ft_vals)}', ha='center', va='bottom',
                fontsize=6, color='gray', transform=ax.get_xaxis_transform())

        y_lo = min(y_lo, bl_mu - (bl_ci or 0.0), ft_mu - (ft_ci or 0.0))
        y_hi = max(y_hi, bl_mu + (bl_ci or 0.0), ft_mu + (ft_ci or 0.0))

    divider_x = (group_ranges[0][1] + group_ranges[0][2] + group_ranges[1][1] - group_ranges[1][2]) / 2
    ax.axvline(divider_x, color='#999999', linewidth=1.0, zorder=1)

    ax.set_xticks([x for _, x in all_items])
    ax.set_xticklabels([point_label(k, show_ep) for k, _ in all_items], fontsize=7)
    ax.tick_params(axis='x', pad=3)

    trans = ax.get_xaxis_transform()
    bracket_y, label_y = -0.30, -0.36
    for vk, gxc, ghalf in group_ranges:
        ax.plot([gxc - ghalf, gxc + ghalf], [bracket_y, bracket_y],
                transform=trans, color='#555', linewidth=0.9, clip_on=False, zorder=2)
        ax.annotate(VIEW_LABEL[vk], xy=(gxc, label_y), xycoords=trans,
                    ha='center', va='top', fontsize=8, color='#222', annotation_clip=False)

    x_lo = group_ranges[0][1] - group_ranges[0][2]
    x_hi = group_ranges[1][1] + group_ranges[1][2]
    ax.set_xlim(x_lo - 0.1, x_hi + 0.1)

    if y_hi > y_lo:
        pad = max((y_hi - y_lo) * 0.15, 0.01)
        ax.set_ylim(max(0.0, y_lo - pad), min(1.02, y_hi + pad))
    else:
        ax.text(0.5, 0.5, 'no qualifying configuration\n(needs ≥%d seeds + a baseline run)' % MIN_SEEDS,
                ha='center', va='center', transform=ax.transAxes, fontsize=9, color='gray')
        ax.set_xticks([])

    ax.set_ylabel(METRIC_LABEL.get(metric, metric), fontsize=9)
    ax.set_title(label, fontsize=10, fontweight='bold', loc='left')
    ax.yaxis.grid(True, linewidth=0.4, alpha=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.spines[['top', 'right']].set_visible(False)


def build_legend(fig, core_colors):
    core_handles = [
        Line2D([0], [0], marker='o', linestyle='none', markerfacecolor=c,
               markeredgecolor='#333', markersize=8, label=humanize_window(core))
        for core, c in core_colors.items()
    ]
    baseline_handles = [
        Line2D([0], [0], marker='o', linestyle='-', color=CONNECTOR_COLOR,
               markerfacecolor=BASELINE_FACE, markeredgecolor=BASELINE_EDGE, markersize=8,
               label='random-init baseline'),
    ]

    leg1 = fig.legend(handles=core_handles, loc='lower center', bbox_to_anchor=(0.35, -0.02),
                       ncol=min(4, len(core_handles)) or 1, fontsize=8, frameon=False,
                       title='Window config', title_fontsize=8.5)
    fig.legend(handles=baseline_handles, loc='lower center', bbox_to_anchor=(0.75, -0.02),
               ncol=1, fontsize=8, frameon=False, title='Reference', title_fontsize=8.5)
    fig.add_artist(leg1)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    global MIN_SEEDS
    parser = argparse.ArgumentParser()
    parser.add_argument('--pairs', nargs='+', default=[p[0] for p in DATASET_PAIRS],
                        choices=[p[0] for p in DATASET_PAIRS],
                        help='Which dataset pairs to plot (default: all)')
    parser.add_argument('--metric', default=DEFAULT_METRIC, choices=list(METRIC_LABEL),
                        help='Default metric for pairs without a PAIR_METRIC / --metric-override entry')
    parser.add_argument('--metric-override', nargs='+', default=[], metavar='PAIR=METRIC',
                        help='Per-pair metric override, e.g. --metric-override sleepeeg_epilepsy=f1_macro')
    parser.add_argument('--min-seeds', type=int, default=MIN_SEEDS,
                        help='Minimum finetune seeds required for a configuration to be shown')
    parser.add_argument('--out', default='plots/dataset_pairs_results.pdf',
                        help='Output file (pdf/png/svg)')
    args = parser.parse_args()
    MIN_SEEDS = args.min_seeds

    pair_metric = dict(PAIR_METRIC)
    for item in args.metric_override:
        key, _, metric = item.partition('=')
        if key not in PAIR_BY_KEY or metric not in METRIC_LABEL:
            parser.error(f'invalid --metric-override entry: {item!r}')
        pair_metric[key] = metric

    panels = []
    for key in args.pairs:
        pair_key, pretrain_tag, finetune_tag, label = PAIR_BY_KEY[key]
        data = load_pair(pretrain_tag, finetune_tag)
        if data is None:
            print(f'  Skipping {label}: no {finetune_tag} results yet')
            continue
        if not data:
            print(f'  Skipping {label}: no configuration with ≥{MIN_SEEDS} seeds + a baseline run')
            continue
        metric = pair_metric.get(pair_key, args.metric)
        panels.append((label, data, metric))

    if not panels:
        print('No data found for any requested dataset pair.')
        return

    all_items, group_ranges, show_ep, core_colors = build_layout([data for _label, data, _metric in panels])

    nrows = len(panels)
    fig_height = 2.6 * nrows + 1.0
    fig, axes = plt.subplots(nrows, 1, figsize=(11.5, fig_height), squeeze=False)

    for idx, (label, data, metric) in enumerate(panels):
        plot_pair(axes[idx][0], label, data, metric,
                  all_items, group_ranges, show_ep, core_colors)

    build_legend(fig, core_colors)
    top_margin = 0.6 / fig_height     # reserve ~0.6in for the suptitle regardless of nrows
    bottom_margin = 0.5 / fig_height  # reserve ~0.5in for the bottom legend
    fig.suptitle(f'Multi-View Contrastive Learning — dx+xf vs. logsig+xf '
                 f'(≥{MIN_SEEDS} seeds, baseline-matched)',
                 fontsize=13, fontweight='bold', y=1.0)
    fig.subplots_adjust(hspace=1.0, top=1 - top_margin, bottom=bottom_margin)

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    plt.savefig(args.out, bbox_inches='tight', dpi=150)
    base, _ = os.path.splitext(args.out)
    if not args.out.endswith('.png'):
        plt.savefig(base + '.png', bbox_inches='tight', dpi=150)
    print(f'Saved → {args.out}')
    print(f'Dataset pairs plotted: {[p[0] for p in panels]}')


if __name__ == '__main__':
    main()

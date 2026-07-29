"""
visualize_epoch_ablation.py — 200/100 vs 2/10 epoch comparison, all pretrain
sources.

For each (pretrain dataset, finetune target, view, window, encoder,
channel_adapt) configuration, compares the full run (epochs_pretrain=200,
epochs_finetune=100) against the quick run (epochs_pretrain=2,
epochs_finetune=10) — only where BOTH exist for the exact same configuration.
Scans every out_finetune/*/final_test_metric_summary.tsv, so this covers all
pretrain sources (capture24, capture24mini, SleepEEG, HARTH, ...), not just
capture24. (pretrain, target) pairs with no such config pair are skipped.

Usage: python visualize_epoch_ablation.py
"""

import glob
import os
import re
import sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from visualize_capture24 import (
    VIEW_LABEL, ENC_AXIS_LABEL, MAJOR_METHOD, ADJUSTMENT, METHOD_LABEL, ADJUSTMENT_LABEL,
)

EP_GROUPS = [200, 2]
EP_LABEL = {200: 'full (pretrain 200ep / finetune 100ep)',
            2:   'quick (pretrain 2ep / finetune 10ep)'}
# validated categorical slots 1 (blue) and 8 (orange) — two most CVD-separated hues.
EP_COLOR = {200: '#2a78d6', 2: '#eb6834'}

PRETRAIN_LABEL = {
    '_DA_capture24_256_00':     'capture24',
    '_DA_capture24mini_256_00': 'capture24-mini',
    '_DA_SleepEEG_256_00':      'SleepEEG',
    '_DA_HARTH_256_00':         'HARTH',
}
_TARGET_OVERRIDES = {'USC_HAD': 'USC-HAD'}


def _target_label(tag_dir: str) -> str:
    """'_DA_USC_HAD_256_00' -> 'USC-HAD', '_DA_HAR70plus_256_00' -> 'HAR70plus'."""
    short = re.sub(r'^_DA_|_256_00$', '', tag_dir)
    return _TARGET_OVERRIDES.get(short, short)


def _pretrain_label(tag: str) -> str:
    return PRETRAIN_LABEL.get(tag, re.sub(r'^_DA_|_256_00$', '', tag))


# short channel-adapt labels for x-tick text (CHADAPT_LABEL's "(x overlay)" suffix
# only makes sense next to an actual overlay mark, which this plot doesn't draw)
_CHADAPT_SHORT = {'drop': 'drop', 'pca': 'pca', 'copy': 'copy'}


def _config_label(view_key, window, enc, ca):
    view = VIEW_LABEL[view_key].replace('\n', ' ')
    method = METHOD_LABEL[MAJOR_METHOD[window]]
    adj = ADJUSTMENT[window]
    win_label = method if adj == 'none' else f'{method} ({ADJUSTMENT_LABEL[adj]})'
    label = f'{view} · {win_label} · {ENC_AXIS_LABEL[enc]}'
    if ca != 'none':
        label += f' · {_CHADAPT_SHORT[ca]}'
    return label


def parse_finetune_row(name: str):
    """Return (pretrain_tag, view_key, window, encoder, channel_adapt, epochs_pretrain)
    for a finetune run_name, else None. Unlike visualize_capture24.parse_finetune_row,
    this is not scoped to any single pretrain source."""
    if not name or not name.endswith('_finetune'):
        return None
    pt_m = re.search(r'_pt-(.+?)_v2', name)
    if not pt_m:
        return None
    pretrain = pt_m.group(1)
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

    m = re.search(r'(v2[a-z]+(?:_(?:v3[a-z]+|nview))?)_ep', name)
    if not m:
        return None
    view_key = m.group(1)
    if '_v3' not in view_key and not view_key.endswith('_nview'):
        view_key = view_key + '_nview'
    if view_key not in VIEW_LABEL:
        return None

    has_lead_lag = re.search(r'_ll\d+', name) is not None
    if '_win64_norm' in name:
        if has_lead_lag:
            window = 'win64_norm_ll'
        elif '_lag' in name:
            window = 'norm_win64_lag'
        else:
            window = 'norm_win64'
    else:
        window = 'global'
        for w in ('win128_s7', 'tukey128_s7', 'win64', 'win128', 'tukey64', 'tukey128'):
            if f'_{w}_' in name or f'_{w}.' in name:
                window = w
                break
        lsn_m = re.search(r'_lsn([\d.]+)_', name)
        if lsn_m:
            window = f'{window}_lsn{lsn_m.group(1)}'
        if '_sl1' in name:
            window = window + '_sl1'
        if has_lead_lag:
            window = window + '_ll'
        if re.search(r'_rp[\d.]+', name):
            window = window + '_rp'

    if window not in MAJOR_METHOD:
        return None

    ca_match = re.search(r'_cadrop\d*|_capca\d*|_caexpand\d*', name)
    if ca_match:
        g = ca_match.group(0)
        channel_adapt = 'drop' if 'cadrop' in g else ('pca' if 'capca' in g else 'copy')
    else:
        channel_adapt = 'none'

    ep_m = re.search(r'_ep(\d+)_', name)
    if not ep_m:
        return None
    epochs_pretrain = int(ep_m.group(1))

    return pretrain, view_key, window, encoder, channel_adapt, epochs_pretrain


def _mean_ci(vals):
    arr = np.array(vals, dtype=float)
    mu = arr.mean()
    if len(arr) >= 2:
        se = stats.sem(arr)
        ci = se * stats.t.ppf(0.975, len(arr) - 1)
    else:
        ci = 0.0
    return mu, ci


def load_paired_data():
    """Return {(pretrain_tag, target_dir): {config: {ep: [scores]}}} restricted
    to configs with data in both EP_GROUPS, across every out_finetune dir."""
    raw = {}  # (pretrain, target) -> config -> ep -> [scores]
    for path in sorted(glob.glob('out_finetune/*/final_test_metric_summary.tsv')):
        target_dir = path.split(os.sep)[1]
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
                pretrain, view_key, window, enc, ca, ep = parsed
                if ep not in EP_GROUPS:
                    continue
                try:
                    score = float(parts[1])
                except ValueError:
                    continue
                config = (view_key, window, enc, ca)
                key = (pretrain, target_dir)
                raw.setdefault(key, {}).setdefault(config, {}).setdefault(ep, []).append(score)

    result = {}
    for key, by_config in raw.items():
        paired = {c: eps for c, eps in by_config.items()
                  if all(ep in eps for ep in EP_GROUPS)}
        if paired:
            result[key] = paired
    return result


def main():
    data = load_paired_data()
    if not data:
        print('No configuration has data for both 200/100 and 2/10 epoch runs.')
        return

    # stable order: group by target, then pretrain
    keys = sorted(data.keys(), key=lambda k: (_target_label(k[1]), _pretrain_label(k[0])))

    n = len(keys)
    max_configs = max(len(data[k]) for k in keys)
    fig, axes = plt.subplots(n, 1, squeeze=False,
                              figsize=(max(7, 1.7 * max_configs), 4.0 * n))
    axes = axes[:, 0]

    for ax, key in zip(axes, keys):
        pretrain, target_dir = key
        paired = data[key]
        configs = sorted(paired.keys())
        x = np.arange(len(configs))
        width = 0.28

        for gi, ep in enumerate(EP_GROUPS):
            offset = (gi - 0.5) * width
            for xi, config in enumerate(configs):
                vals = paired[config][ep]
                mu, ci = _mean_ci(vals)
                yerr = None if ci is None else [[min(ci, mu)], [min(ci, 1.0 - mu)]]
                px = x[xi] + offset
                ax.errorbar(px, mu, yerr=yerr,
                            fmt='o', markersize=7,
                            markerfacecolor=EP_COLOR[ep], markeredgecolor='#333',
                            markeredgewidth=0.7, ecolor='#333', elinewidth=1.3,
                            capsize=3, zorder=5)
                ax.annotate(f'n={len(vals)}', xy=(px, mu), xytext=(0, -11),
                            textcoords='offset points', ha='center', va='top',
                            fontsize=6.5, color='#555', zorder=6,
                            annotation_clip=False)

        for xi in range(len(configs) - 1):
            ax.axvline(xi + 0.5, color='#cccccc', linewidth=0.6, linestyle=':', zorder=1)

        ax.set_xticks(x)
        ax.set_xticklabels([_config_label(*c) for c in configs],
                            fontsize=8, rotation=15, ha='right')
        ax.set_ylabel('Test accuracy', fontsize=9)
        ax.set_title(f'{_pretrain_label(pretrain)} → {_target_label(target_dir)}',
                     fontsize=10, fontweight='bold')
        ax.yaxis.grid(True, linewidth=0.4, alpha=0.5, zorder=0)
        ax.set_axisbelow(True)
        ax.spines[['top', 'right']].set_visible(False)
        ax.set_xlim(-0.5, len(configs) - 0.5)

    handles = [Line2D([0], [0], marker='o', linestyle='none', markersize=8,
                       markerfacecolor=EP_COLOR[ep], markeredgecolor='#333',
                       label=EP_LABEL[ep])
               for ep in EP_GROUPS]
    # anchor to the last axes (not the whole figure) so the legend sits a fixed
    # fraction below the final panel's rotated tick labels regardless of n rows
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, -0.55),
               bbox_transform=axes[-1].transAxes,
               ncol=2, fontsize=9, frameon=False,
               title='Training budget', title_fontsize=9.5)

    fig.suptitle('Multi-View Contrastive Learning — 200/100 vs 2/10 epoch comparison (95% CI)',
                 fontsize=12, fontweight='bold', y=1.01)
    # default top margin is a fixed *fraction* of figure height, so it grows into a
    # large blank band in inches once the figure gets tall (many panels) — pin it
    # to a small fraction instead of leaving it at matplotlib's default.
    fig.subplots_adjust(top=0.97, hspace=0.6)

    os.makedirs('plots', exist_ok=True)
    out = 'plots/epoch_ablation.pdf'
    plt.savefig(out, bbox_inches='tight', dpi=150)
    plt.savefig(out.replace('.pdf', '.png'), bbox_inches='tight', dpi=150)
    print(f'Saved -> {out}')
    print('Pairs plotted:', [f'{_pretrain_label(p)}→{_target_label(t)}' for p, t in keys])


if __name__ == '__main__':
    main()

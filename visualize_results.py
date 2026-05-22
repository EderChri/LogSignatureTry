"""
visualize_results.py — Finetune accuracy across all logsig configurations.

Two panels side-by-side: Epilepsy (SleepEEG pretrain) | HAR70plus (HARTH pretrain).

X-axis groups:  view combination  (dx+xf | logsig+xf | dx+logsig | logsig 2-view)
Within groups:  window variant sub-groups × encoder bars
  global sub-group: 3 bars  [Transformer (mean) | Transformer (last/plast) | MLP-LogSig]
  windowed sub-groups: 2 bars  [Transformer | MLP-LogSig]
Colour:         window variant
Hatch:          encoder / pooling variant
Reference line: dx+xf (Transformer, global) score shown as dotted line per panel.

Usage: python visualize_results.py
"""

import re
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats

matplotlib.rcParams['hatch.linewidth'] = 2.0   # default 1.0 is invisible on thin bars

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VIEW_ORDER = ['v2dx_v3xf', 'v2logsig_v3xf', 'v2dx_v3logsig', 'v2logsig_nview']
VIEW_LABEL = {
    'v2dx_v3xf':      'dx + xf\n(baseline)',
    'v2logsig_v3xf':  'logsig + xf',
    'v2dx_v3logsig':  'dx + logsig',
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

# Three encoder variants:
#   transformer       — default, mean pool over logsig time dim (auto)
#   transformer_plast — ablation: last-token pool (--logsig_pool last)
#   mlp_logsig        — MLP branch, last-token pool (auto for stream)
ENCODERS  = ['transformer', 'transformer_plast', 'mlp_logsig',
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
    'mlp_logsig':           'MLP-LogSig (last pool)',
    'mlp_logsig_bilinear':  'MLP-LogSig bilinear',
    'transformer_bilinear': 'Transformer bilinear',
}

# ---------------------------------------------------------------------------
# Bar geometry
# ---------------------------------------------------------------------------

BAR_W     = 0.048   # width of each individual bar
INNER_GAP = 0.008   # gap between bars within a sub-group
WIN_GAP   = 0.025   # gap between window sub-groups
VIEW_SEP  = 1.20    # centre-to-centre distance between view groups

SUB_W_2 = 2 * BAR_W + INNER_GAP        # 2-bar sub-group (windowed, no bilinear)
SUB_W_3 = 3 * BAR_W + 2 * INNER_GAP   # 3-bar sub-group (global or bilinear windowed)
_N_WIN = len(WINDOW_ORDER)             # total number of window sub-groups
# win64, win128, tukey128 have bilinear data → 3-bar; others → 2-bar
_BIL_WIN_SET = {'win64', 'win128', 'tukey128'}
LOGSIG_SPAN = (
    SUB_W_3                                                           # global
    + sum(SUB_W_3 if w in _BIL_WIN_SET else SUB_W_2
          for w in WINDOW_ORDER[1:])
    + (_N_WIN - 1) * WIN_GAP
)


def _view_bar_specs(view_key):
    """Return [(window, encoder, rel_x), ...] relative to the group centre."""
    if view_key == 'v2dx_v3xf':
        # 2 bars: transformer (standard) | transformer (bilinear)
        _hw = BAR_W / 2 + INNER_GAP / 2
        return [
            ('global', 'transformer',          -_hw),
            ('global', 'transformer_bilinear',  +_hw),
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
            # 3 bars: transformer (mean) | transformer_plast (last) | mlp_logsig
            specs.append((win, 'transformer',       xc - BAR_W - INNER_GAP))
            specs.append((win, 'transformer_plast', xc))
            specs.append((win, 'mlp_logsig',        xc + BAR_W + INNER_GAP))
        elif is_bil:
            # 3 bars: transformer | mlp_logsig | mlp_logsig_bilinear
            specs.append((win, 'transformer',          xc - BAR_W - INNER_GAP))
            specs.append((win, 'mlp_logsig',           xc))
            specs.append((win, 'mlp_logsig_bilinear',  xc + BAR_W + INNER_GAP))
        else:
            # 2 bars: transformer | mlp_logsig
            specs.append((win, 'transformer', xc - BAR_W / 2 - INNER_GAP / 2))
            specs.append((win, 'mlp_logsig',  xc + BAR_W / 2 + INNER_GAP / 2))

        x += sw

    return specs


_BAR_SPECS = {v: _view_bar_specs(v) for v in VIEW_ORDER}

# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def parse_finetune_row(name: str):
    """
    Returns (view_key, window, encoder) for finetune rows, else None.
    Skips probe rows, non-finetune suffixes, cross-time and view-embed variants.
    Bilinear variants are included (mlp_logsig_bilinear / transformer_bilinear).
    """
    if not name or not name.endswith('_finetune') or name.startswith('probe_'):
        return None
    # Cross-time and view-embed interaction variants belong to a separate plot.
    if any(m in name for m in ('_ilcrosstime_', '_ilviewembed_')):
        return None

    is_bilinear = '_ilbilinear_' in name
    # Check mlp_logsig before plast (plast is transformer-only)
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


# ---------------------------------------------------------------------------
# Interaction-type constants
# ---------------------------------------------------------------------------

IL_ORDER = ['attention', 'bilinear', 'cross_time', 'view_embed']
IL_LABEL = {'attention': 'Standard', 'bilinear': 'Bilinear',
            'cross_time': 'Cross-time', 'view_embed': 'View-embed'}
IL_COLOR = {'attention': '#4C72B0', 'bilinear': '#DD8452',
            'cross_time': '#55A868', 'view_embed': '#C44E52'}
IL_MODE_ORDER = ['finetune', 'freeze', 'baseline']
IL_MODE_HATCH = {'finetune': '', 'freeze': '\\' * 4, 'baseline': '/' * 4}
IL_MODE_LABEL = {'finetune': 'finetune', 'freeze': 'freeze', 'baseline': 'baseline'}
IL_MODE_ALPHA = {'finetune': 1.0, 'freeze': 0.7, 'baseline': 0.5}


def parse_interaction_row(name: str):
    """Returns (il_type, ep_group, mode) for dx+xf interaction-type runs, else None."""
    if not name:
        return None
    if name.endswith('_finetune'):
        mode = 'finetune'
    elif name.endswith('_freeze'):
        mode = 'freeze'
    elif name.endswith('_baseline'):
        mode = 'baseline'
    else:
        return None
    if 'v2dx_v3xf' not in name:
        return None
    ep_group = 'ep200' if '_ep200_' in name else ('ep2' if '_ep2_' in name else None)
    if ep_group is None:
        return None
    if '_ilbilinear_' in name:
        il = 'bilinear'
    elif '_ilcrosstime_' in name:
        il = 'cross_time'
    elif '_ilviewembed_' in name:
        il = 'view_embed'
    else:
        il = 'attention'
    return il, ep_group, mode


def read_interaction_tsv(path: str):
    """Return dict (il_type, ep_group, mode) → list[float]."""
    scores = {}
    try:
        with open(path) as f:
            for line in f:
                line = line.rstrip('\n')
                if not line or line.startswith('#'):
                    continue
                parts = line.split('\t')
                if len(parts) < 2:
                    continue
                parsed = parse_interaction_row(parts[0])
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


def read_tsv(path: str):
    """Return dict (view_key, window, encoder) → list[float] for finetune rows."""
    scores = {}
    try:
        with open(path) as f:
            for line in f:
                line = line.rstrip('\n')
                if not line or line.startswith('#'):
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
# Bilinear vs standard parsing (Figure 3)
# ---------------------------------------------------------------------------

BIL_VIEWS   = ['v2dx_v3logsig', 'v2logsig_v3xf']
BIL_VIEW_LABEL = {
    'v2dx_v3logsig': 'dx + logsig',
    'v2logsig_v3xf': 'logsig + xf',
}
BIL_WINDOWS = ['win64', 'win128', 'tukey128']
BIL_IL      = ['standard', 'bilinear']
BIL_IL_HATCH = {'standard': '', 'bilinear': 'xx'}
BIL_IL_LABEL = {'standard': 'Standard (mlp_logsig)', 'bilinear': 'Bilinear (mlp_logsig)'}


def parse_bilinear_row(name: str):
    """
    Returns (view_key, window, il) for mlp_logsig finetune rows on logsig views, else None.
    il is 'bilinear' if _ilbilinear_ present, else 'standard'.
    """
    if not name or not name.endswith('_finetune'):
        return None
    if '_mlp_logsig_' not in name:
        return None
    m = re.search(r'(v2[a-z]+_v3[a-z]+)_ep', name)
    if not m or m.group(1) not in BIL_VIEWS:
        return None
    view_key = m.group(1)
    il = 'bilinear' if '_ilbilinear_' in name else 'standard'
    window = None
    for w in BIL_WINDOWS:
        if f'_{w}_' in name or f'_{w}.' in name:
            window = w
            break
    if window is None:
        return None
    return view_key, window, il


def read_bilinear_tsv(path: str):
    """Return dict (view_key, window, il) → list[float]."""
    scores = {}
    try:
        with open(path) as f:
            for line in f:
                line = line.rstrip('\n')
                if not line or line.startswith('#'):
                    continue
                parts = line.split('\t')
                if len(parts) < 2:
                    continue
                parsed = parse_bilinear_row(parts[0])
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
# Load data
# ---------------------------------------------------------------------------

epilepsy_scores = read_tsv(
    'out_finetune/_DA_Epilepsy_256_00/final_test_metric_summary.tsv')
har_scores = read_tsv(
    'out_finetune/_DA_HAR70plus_256_00/final_test_metric_summary.tsv')

panels = [
    ('Epilepsy  (SleepEEG pretrain)', epilepsy_scores),
    ('HAR70plus  (HARTH pretrain)',   har_scores),
]

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

fig, axes = plt.subplots(1, 2, figsize=(22, 6.5), sharey=False)
fig.suptitle('Finetune accuracy — view combinations, logsig variants, encoders',
             fontsize=13, fontweight='bold', y=1.01)

view_centres = np.arange(len(VIEW_ORDER)) * VIEW_SEP

for ax, (title, scores) in zip(axes, panels):

    ref_vals  = scores.get(('v2dx_v3xf', 'global', 'transformer'), [])
    ref_score = np.mean(ref_vals) if ref_vals else None

    for vi, view_key in enumerate(VIEW_ORDER):
        xc = view_centres[vi]
        specs = _BAR_SPECS[view_key]

        for win, enc, rel_x in specs:
            key  = (view_key, win, enc)
            vals = scores.get(key, [])
            if not vals:
                continue
            score = np.mean(vals)

            color = WINDOW_COLOR[win]
            hatch = ENC_HATCH[enc]
            ax.bar(xc + rel_x, score,
                   width=BAR_W - 0.005,
                   color=color,
                   hatch=hatch,
                   edgecolor='white' if hatch == '' else '#333',
                   linewidth=0.5,
                   zorder=3)
            if len(vals) > 1:
                n = len(vals)
                std = np.std(vals, ddof=1)
                ci95 = stats.t.ppf(0.975, df=n - 1) * std / np.sqrt(n)
                min_val, max_val = min(vals), max(vals)
                # min/max range (thin grey whiskers)
                ax.errorbar(xc + rel_x, score,
                            yerr=[[score - min_val], [max_val - score]],
                            fmt='none',
                            ecolor='#999',
                            elinewidth=0.6,
                            capsize=3,
                            zorder=4)
                # 95% CI (thicker black bar on top)
                ax.errorbar(xc + rel_x, score,
                            yerr=ci95,
                            fmt='none',
                            ecolor='#111',
                            elinewidth=1.6,
                            capsize=2,
                            zorder=5)

        if vi < len(VIEW_ORDER) - 1:
            ax.axvline(xc + VIEW_SEP / 2, color='#cccccc',
                       linewidth=0.8, linestyle='--', zorder=1)

    if ref_score is not None:
        ax.axhline(ref_score, color='#222', linewidth=1.2,
                   linestyle=':', zorder=4)
        ax.text(view_centres[-1] + VIEW_SEP * 0.45,
                ref_score + 0.002,
                f'ref {ref_score:.3f}',
                fontsize=7.5, va='bottom', ha='right', color='#222')

    ax.set_title(title, fontsize=10, pad=8)
    ax.set_xticks(view_centres)
    ax.set_xticklabels([VIEW_LABEL[v] for v in VIEW_ORDER], fontsize=9)
    ax.set_ylabel('Accuracy', fontsize=9)

    all_scores = [np.mean(v) for v in scores.values() if v]
    ymin = max(0.0, min(all_scores) - 0.04) if all_scores else 0.0
    ax.set_ylim(ymin, 1.03)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.2f}'))
    ax.grid(axis='y', linestyle='--', alpha=0.4, zorder=0)
    ax.spines[['top', 'right']].set_visible(False)
    half = LOGSIG_SPAN / 2 + 0.12
    ax.set_xlim(-half, (len(VIEW_ORDER) - 1) * VIEW_SEP + half)

# ---------------------------------------------------------------------------
# Legend
# ---------------------------------------------------------------------------

window_patches = [
    mpatches.Patch(facecolor=WINDOW_COLOR[w], edgecolor='white',
                   label=WINDOW_LABEL[w])
    for w in WINDOW_ORDER
]
enc_patches = [
    mpatches.Patch(facecolor='#888', hatch=ENC_HATCH[e],
                   edgecolor='#333', label=ENC_LABEL[e])
    for e in ['transformer', 'transformer_plast', 'mlp_logsig', 'mlp_logsig_bilinear',
              'transformer_bilinear']
]

leg1 = fig.legend(handles=window_patches,
                  loc='lower center', bbox_to_anchor=(0.28, -0.07),
                  ncol=5, fontsize=8.5, frameon=False,
                  title='Logsig window', title_fontsize=9)
leg2 = fig.legend(handles=enc_patches,
                  loc='lower center', bbox_to_anchor=(0.76, -0.07),
                  ncol=3, fontsize=8.5, frameon=False,
                  title='Encoder / pooling', title_fontsize=9)
fig.add_artist(leg1)

plt.tight_layout()
plt.savefig('finetune_results.pdf', bbox_inches='tight')
plt.savefig('finetune_results.png', dpi=150, bbox_inches='tight')
print('Saved: finetune_results.pdf  finetune_results.png')
plt.show()

# ---------------------------------------------------------------------------
# Figure 2: Interaction-type comparison (dx+xf, ep2 vs ep200)
# ---------------------------------------------------------------------------

il_epilepsy = read_interaction_tsv(
    'out_finetune/_DA_Epilepsy_256_00/final_test_metric_summary.tsv')
il_har = read_interaction_tsv(
    'out_finetune/_DA_HAR70plus_256_00/final_test_metric_summary.tsv')

il_panels = [
    ('Epilepsy  (SleepEEG pretrain) — interaction type', il_epilepsy),
    ('HAR70plus  (HARTH pretrain) — interaction type',   il_har),
]

IL_BAR_W   = 0.12
IL_MODE_GAP = 0.01
IL_GROUP_W  = len(IL_MODE_ORDER) * IL_BAR_W + (len(IL_MODE_ORDER) - 1) * IL_MODE_GAP
IL_GROUP_SEP = 0.65

fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5), sharey=False)
fig2.suptitle('Interaction-type comparison — dx + xf views, ep2 (bars) and ep200 (markers)',
              fontsize=12, fontweight='bold', y=1.02)

for ax, (title, scores) in zip(axes2, il_panels):
    ep_groups = ['ep2', 'ep200']
    ep_marker = {'ep2': None, 'ep200': '*'}
    ep_ms     = {'ep2': 0,    'ep200': 10}

    x_centres = np.arange(len(IL_ORDER)) * IL_GROUP_SEP

    for xi, il in enumerate(IL_ORDER):
        xc = x_centres[xi]

        # ep2 bars (finetune / freeze / baseline)
        for mi, mode in enumerate(IL_MODE_ORDER):
            vals = scores.get((il, 'ep2', mode), [])
            if not vals:
                continue
            score = np.mean(vals)
            bar_x = xc + (mi - 1) * (IL_BAR_W + IL_MODE_GAP)
            ax.bar(bar_x, score,
                   width=IL_BAR_W,
                   color=IL_COLOR[il],
                   hatch=IL_MODE_HATCH[mode],
                   alpha=IL_MODE_ALPHA[mode],
                   edgecolor='white' if IL_MODE_HATCH[mode] == '' else '#333',
                   linewidth=0.5,
                   zorder=3)
            if len(vals) > 1:
                ax.errorbar(bar_x, score,
                            yerr=np.std(vals, ddof=1) / len(vals) ** 0.5,
                            fmt='none', ecolor='#111', elinewidth=1.5,
                            capsize=3, zorder=5)

        # ep200 markers (overlay on finetune bar position)
        for mi, mode in enumerate(IL_MODE_ORDER):
            vals200 = scores.get((il, 'ep200', mode), [])
            if not vals200:
                continue
            score200 = np.mean(vals200)
            bar_x = xc + (mi - 1) * (IL_BAR_W + IL_MODE_GAP)
            ax.plot(bar_x, score200, marker='D', markersize=7,
                    color='black', markerfacecolor=IL_COLOR[il],
                    markeredgewidth=1.2, zorder=6,
                    label=f'{IL_LABEL[il]} ep200' if mi == 0 else '_')

    ax.set_title(title, fontsize=10, pad=8)
    ax.set_xticks(x_centres)
    ax.set_xticklabels([IL_LABEL[il] for il in IL_ORDER], fontsize=10)
    ax.set_ylabel('Accuracy', fontsize=9)
    ax.grid(axis='y', linestyle='--', alpha=0.4, zorder=0)
    ax.spines[['top', 'right']].set_visible(False)

    all_vals = [v for vals in scores.values() for v in vals]
    if all_vals:
        ax.set_ylim(max(0.0, min(all_vals) - 0.05), 1.03)

# Shared legends
mode_patches = [
    mpatches.Patch(facecolor='#888', hatch=IL_MODE_HATCH[m],
                   edgecolor='#333', alpha=IL_MODE_ALPHA[m], label=IL_MODE_LABEL[m])
    for m in IL_MODE_ORDER
]
ep200_handle = plt.Line2D([0], [0], marker='D', color='w',
                           markerfacecolor='#555', markeredgecolor='black',
                           markersize=8, label='ep200 (marker)')
fig2.legend(handles=mode_patches + [ep200_handle],
            loc='lower center', bbox_to_anchor=(0.5, -0.06),
            ncol=4, fontsize=9, frameon=False,
            title='Mode / epoch', title_fontsize=9)

plt.tight_layout()
plt.savefig('interaction_type_results.pdf', bbox_inches='tight')
plt.savefig('interaction_type_results.png', dpi=150, bbox_inches='tight')
print('Saved: interaction_type_results.pdf  interaction_type_results.png')
plt.show()

# ---------------------------------------------------------------------------
# Figure 3: Bilinear vs Standard — mlp_logsig, logsig view combos
#
# X groups: view combo (dx+logsig | logsig+xf), each split by window variant.
# Each x-position: 2 bars — standard (solid) vs bilinear (cross-hatch).
# Color:  window variant  (matches Figure 1 palette).
# Hatch:  standard = none, bilinear = cross-hatch.
# Error:  thin grey whiskers = min/max range; thick black = 95% CI.
# ---------------------------------------------------------------------------

bil_epilepsy = read_bilinear_tsv(
    'out_finetune/_DA_Epilepsy_256_00/final_test_metric_summary.tsv')
bil_har = read_bilinear_tsv(
    'out_finetune/_DA_HAR70plus_256_00/final_test_metric_summary.tsv')

bil_panels = [
    ('Epilepsy  (SleepEEG pretrain)', bil_epilepsy, epilepsy_scores),
    ('HAR70plus  (HARTH pretrain)',   bil_har,       har_scores),
]

BIL_BAR_W    = 0.12
BIL_PAIR_GAP = 0.02   # gap between the two bars of a pair
BIL_WIN_GAP  = 0.08   # gap between window pairs within a view group
BIL_VIEW_SEP = 1.20   # centre-to-centre distance between view groups

# positions within a (view, window) group: standard bar left, bilinear bar right
_pair_half = (BIL_BAR_W + BIL_PAIR_GAP) / 2
BIL_IL_OFFSET = {'standard': -_pair_half, 'bilinear': +_pair_half}

# x-centre offsets for the 3 window pairs within a view group
_n_win   = len(BIL_WINDOWS)
_win_span = (_n_win - 1) * (2 * BIL_BAR_W + BIL_PAIR_GAP + BIL_WIN_GAP)
BIL_WIN_OFFSETS = {
    w: -_win_span / 2 + i * (2 * BIL_BAR_W + BIL_PAIR_GAP + BIL_WIN_GAP)
    for i, w in enumerate(BIL_WINDOWS)
}

fig3, axes3 = plt.subplots(1, 2, figsize=(14, 5.5), sharey=False)
fig3.suptitle(
    'Bilinear vs Standard interaction layer — mlp_logsig encoder, logsig views',
    fontsize=12, fontweight='bold', y=1.02)

bil_view_centres = np.arange(len(BIL_VIEWS)) * BIL_VIEW_SEP

for ax, (title, scores, ref_scores) in zip(axes3, bil_panels):

    has_data = False

    for vi, view_key in enumerate(BIL_VIEWS):
        xc = bil_view_centres[vi]

        for win in BIL_WINDOWS:
            wx = xc + BIL_WIN_OFFSETS[win]
            color = WINDOW_COLOR[win]

            for il in BIL_IL:
                key  = (view_key, win, il)
                vals = scores.get(key, [])
                bx   = wx + BIL_IL_OFFSET[il]

                if vals:
                    has_data = True
                    score = np.mean(vals)
                    ax.bar(bx, score,
                           width=BIL_BAR_W,
                           color=color,
                           hatch=BIL_IL_HATCH[il],
                           edgecolor='white' if BIL_IL_HATCH[il] == '' else '#333',
                           linewidth=0.5,
                           zorder=3)
                    if len(vals) > 1:
                        n   = len(vals)
                        std = np.std(vals, ddof=1)
                        ci95 = stats.t.ppf(0.975, df=n - 1) * std / np.sqrt(n)
                        ax.errorbar(bx, score,
                                    yerr=[[score - min(vals)], [max(vals) - score]],
                                    fmt='none', ecolor='#aaa', elinewidth=0.6,
                                    capsize=3, zorder=4)
                        ax.errorbar(bx, score,
                                    yerr=ci95,
                                    fmt='none', ecolor='#111', elinewidth=1.6,
                                    capsize=2, zorder=5)
                else:
                    # placeholder so the chart isn't silent about missing data
                    ax.bar(bx, 0,
                           width=BIL_BAR_W,
                           color=color,
                           hatch=BIL_IL_HATCH[il],
                           edgecolor='#ccc',
                           linewidth=0.5,
                           alpha=0.3,
                           zorder=3)

        # vertical separator between view groups
        if vi < len(BIL_VIEWS) - 1:
            ax.axvline(xc + BIL_VIEW_SEP / 2,
                       color='#cccccc', linewidth=0.8, linestyle='--', zorder=1)

    # dx+xf baseline reference line + CI band
    ref_vals = ref_scores.get(('v2dx_v3xf', 'global', 'transformer'), [])
    ref = np.mean(ref_vals) if ref_vals else None
    if ref is not None:
        if len(ref_vals) > 1:
            n_ref = len(ref_vals)
            std_ref = np.std(ref_vals, ddof=1)
            ci95_ref = stats.t.ppf(0.975, df=n_ref - 1) * std_ref / np.sqrt(n_ref)
            # min/max range — very light band
            ax.axhspan(min(ref_vals), max(ref_vals),
                       color='#888', alpha=0.10, linewidth=0, zorder=2,
                       label='dx+xf range')
            # 95% CI band
            ax.axhspan(ref - ci95_ref, ref + ci95_ref,
                       color='#444', alpha=0.22, linewidth=0, zorder=3,
                       label='dx+xf 95% CI')
            label_text = f'dx+xf ref {ref:.3f} ±{ci95_ref:.3f}'
        else:
            label_text = f'dx+xf ref {ref:.3f}'
        ax.axhline(ref, color='#222', linewidth=1.2, linestyle=':', zorder=4)
        ax.text(bil_view_centres[-1] + BIL_VIEW_SEP * 0.45,
                ref + 0.002,
                label_text,
                fontsize=7.5, va='bottom', ha='right', color='#222')

    ax.set_title(title, fontsize=10, pad=8)
    ax.set_xticks(bil_view_centres)
    ax.set_xticklabels([BIL_VIEW_LABEL[v] for v in BIL_VIEWS], fontsize=10)
    ax.set_ylabel('Accuracy', fontsize=9)
    ax.grid(axis='y', linestyle='--', alpha=0.4, zorder=0)
    ax.spines[['top', 'right']].set_visible(False)

    all_vals = [v for vals in scores.values() for v in vals]
    if ref is not None:
        all_vals.append(ref)
    if all_vals:
        ax.set_ylim(max(0.0, min(all_vals) - 0.04), 1.03)

    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.2f}'))

    half = _win_span / 2 + BIL_BAR_W + 0.10
    ax.set_xlim(-half, (len(BIL_VIEWS) - 1) * BIL_VIEW_SEP + half)

# Add minor x-tick labels for window variants within each view group
for ax in axes3:
    for vi, view_key in enumerate(BIL_VIEWS):
        xc = bil_view_centres[vi]
        for win in BIL_WINDOWS:
            wx = xc + BIL_WIN_OFFSETS[win]
            ax.annotate(WINDOW_LABEL[win],
                        xy=(wx, 0), xycoords=('data', 'axes fraction'),
                        xytext=(0, -18), textcoords='offset points',
                        ha='center', va='top', fontsize=7.5, color='#555')

# Legend
bil_win_patches = [
    mpatches.Patch(facecolor=WINDOW_COLOR[w], edgecolor='white', label=WINDOW_LABEL[w])
    for w in BIL_WINDOWS
]
bil_il_patches = [
    mpatches.Patch(facecolor='#888', hatch=BIL_IL_HATCH[il],
                   edgecolor='#333', label=BIL_IL_LABEL[il])
    for il in BIL_IL
]
ref_ci_patch   = mpatches.Patch(facecolor='#444', alpha=0.22, linewidth=0,
                                 label='dx+xf 95% CI')
ref_range_patch = mpatches.Patch(facecolor='#888', alpha=0.10, linewidth=0,
                                  label='dx+xf min/max range')
fig3.legend(handles=bil_win_patches,
            loc='lower center', bbox_to_anchor=(0.25, -0.06),
            ncol=3, fontsize=8.5, frameon=False,
            title='Logsig window', title_fontsize=9)
fig3.legend(handles=bil_il_patches,
            loc='lower center', bbox_to_anchor=(0.63, -0.06),
            ncol=2, fontsize=8.5, frameon=False,
            title='Interaction layer', title_fontsize=9)
fig3.legend(handles=[ref_ci_patch, ref_range_patch],
            loc='lower center', bbox_to_anchor=(0.90, -0.06),
            ncol=1, fontsize=8.5, frameon=False,
            title='Baseline (dx+xf)', title_fontsize=9)

plt.tight_layout()
plt.savefig('bilinear_logsig_results.pdf', bbox_inches='tight')
plt.savefig('bilinear_logsig_results.png', dpi=150, bbox_inches='tight')
print('Saved: bilinear_logsig_results.pdf  bilinear_logsig_results.png')
plt.show()

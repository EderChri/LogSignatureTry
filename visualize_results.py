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
ENCODERS  = ['transformer', 'transformer_plast', 'mlp_logsig']
ENC_HATCH = {
    'transformer':       '',
    'transformer_plast': '\\' * 6,
    'mlp_logsig':        '/' * 6,
}
ENC_LABEL = {
    'transformer':       'Transformer (mean pool)',
    'transformer_plast': 'Transformer (last pool)',
    'mlp_logsig':        'MLP-LogSig (last pool)',
}

# ---------------------------------------------------------------------------
# Bar geometry
# ---------------------------------------------------------------------------

BAR_W     = 0.048   # width of each individual bar
INNER_GAP = 0.008   # gap between bars within a sub-group
WIN_GAP   = 0.025   # gap between window sub-groups
VIEW_SEP  = 1.20    # centre-to-centre distance between view groups

SUB_W_2 = 2 * BAR_W + INNER_GAP        # 2-bar sub-group (windowed variants)
SUB_W_3 = 3 * BAR_W + 2 * INNER_GAP   # 3-bar sub-group (global: +plast bar)
_N_WIN = len(WINDOW_ORDER)             # total number of window sub-groups
LOGSIG_SPAN = SUB_W_3 + (_N_WIN - 1) * SUB_W_2 + (_N_WIN - 1) * WIN_GAP


def _view_bar_specs(view_key):
    """Return [(window, encoder, rel_x), ...] relative to the group centre."""
    if view_key == 'v2dx_v3xf':
        return [('global', 'transformer', 0.0)]

    specs = []
    x = -LOGSIG_SPAN / 2
    for i, win in enumerate(WINDOW_ORDER):
        if i > 0:
            x += WIN_GAP
        sw = SUB_W_3 if win == 'global' else SUB_W_2
        xc = x + sw / 2

        if win == 'global':
            # 3 bars: transformer (mean) | transformer_plast (last) | mlp_logsig
            specs.append((win, 'transformer',       xc - BAR_W - INNER_GAP))
            specs.append((win, 'transformer_plast', xc))
            specs.append((win, 'mlp_logsig',        xc + BAR_W + INNER_GAP))
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
    Skips probe rows and any non-finetune suffix.
    """
    if not name or not name.endswith('_finetune') or name.startswith('probe_'):
        return None

    # Check mlp_logsig before plast (plast is transformer-only)
    if '_mlp_logsig_' in name:
        encoder = 'mlp_logsig'
    elif '_plast' in name:
        encoder = 'transformer_plast'
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
                sem = np.std(vals, ddof=1) / np.sqrt(len(vals))
                ax.errorbar(xc + rel_x, score,
                            yerr=sem,
                            fmt='none',
                            ecolor='#222',
                            elinewidth=0.8,
                            capsize=2,
                            zorder=4)

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
    for e in ENCODERS
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

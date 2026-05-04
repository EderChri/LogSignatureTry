"""
visualize_finetune.py — Finetune-only results heatmap.

Rows:    (view combination, window/logsig variant)
Columns: transformer encoder | MLP-LogSig encoder
Colour:  accuracy (RdYlGn, range tuned to actual score spread)

dx+xf is the reference config (no logsig window); its mlp_logsig cell is blank.

Usage: python visualize_finetune.py
"""

import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

VIEW_ORDER = ['v2dx_v3xf', 'v2logsig_v3xf', 'v2dx_v3logsig', 'v2logsig_nview']
VIEW_LABEL = {
    'v2dx_v3xf':      'dx + xf',
    'v2logsig_v3xf':  'logsig + xf',
    'v2dx_v3logsig':  'dx + logsig',
    'v2logsig_nview': 'logsig (n-view)',
}
VIEW_COLOR = {
    'v2dx_v3xf':      '#4C72B0',
    'v2logsig_v3xf':  '#DD8452',
    'v2dx_v3logsig':  '#55A868',
    'v2logsig_nview': '#C44E52',
}

WINDOW_ORDER  = ['global', 'win64', 'win128', 'tukey64', 'tukey128']
WINDOW_LABEL  = {
    'global':   'global',
    'win64':    'win 64',
    'win128':   'win 128',
    'tukey64':  'tukey 64',
    'tukey128': 'tukey 128',
}

ENCODERS   = ['transformer', 'mlp_logsig']
ENC_LABEL  = {'transformer': 'Transformer', 'mlp_logsig': 'MLP-LogSig'}

# ---------------------------------------------------------------------------
# Parse
# ---------------------------------------------------------------------------

def parse_finetune_row(name: str):
    """Returns (view_key, window, encoder) or None — finetune rows only."""
    if not name or not name.endswith('_finetune') or name.startswith('probe_'):
        return None

    encoder = 'mlp_logsig' if '_mlp_logsig_' in name else 'transformer'

    m = re.search(r'(v2[a-z]+_(?:v3[a-z]+|nview))_ep', name)
    if not m or m.group(1) not in VIEW_LABEL:
        return None
    view_key = m.group(1)

    window = 'global'
    for w in ('win64', 'win128', 'tukey64', 'tukey128'):
        if f'_{w}_' in name:
            window = w
            break

    return view_key, window, encoder


def read_scores(path: str):
    scores = {}
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
            scores[parsed] = score
    return scores


# ---------------------------------------------------------------------------
# Build row/column index
# ---------------------------------------------------------------------------

def build_grid(scores):
    """Return (rows, matrix, mask) where rows = [(view_key, window), ...].

    matrix shape = (n_rows, 2) for columns [transformer, mlp_logsig].
    mask = True where data is absent (blank cell).
    """
    rows = []
    for view in VIEW_ORDER:
        windows = ['global'] if view == 'v2dx_v3xf' else WINDOW_ORDER
        for win in windows:
            rows.append((view, win))

    n = len(rows)
    matrix = np.full((n, 2), np.nan)
    mask   = np.ones((n, 2), dtype=bool)

    for ri, (view, win) in enumerate(rows):
        for ci, enc in enumerate(ENCODERS):
            key = (view, win, enc)
            # dx+xf has no mlp_logsig variant
            if view == 'v2dx_v3xf' and enc == 'mlp_logsig':
                continue
            val = scores.get(key, np.nan)
            matrix[ri, ci] = val
            if not np.isnan(val):
                mask[ri, ci] = False

    return rows, matrix, mask


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def plot_heatmap(scores, title, ax):
    rows, matrix, mask = build_grid(scores)
    n_rows = len(rows)

    # colour range tight around actual data spread
    valid = matrix[~np.isnan(matrix)]
    vmin = max(valid.min() - 0.01, 0.70)
    vmax = min(valid.max() + 0.01, 1.00)

    cmap = plt.cm.RdYlGn
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

    for ri in range(n_rows):
        for ci in range(2):
            val = matrix[ri, ci]
            if np.isnan(val):
                fc = '#e8e8e8'
            else:
                fc = cmap(norm(val))
            rect = plt.Rectangle([ci, n_rows - ri - 1], 1, 1,
                                  facecolor=fc, edgecolor='white', linewidth=1.5)
            ax.add_patch(rect)
            if not np.isnan(val):
                ax.text(ci + 0.5, n_rows - ri - 0.5,
                        f'{val:.4f}',
                        ha='center', va='center',
                        fontsize=8.5, fontweight='bold',
                        color='black' if 0.35 < norm(val) < 0.75 else 'white')

    # row group separators and left-side view labels
    prev_view = None
    group_starts = {}   # view → first y position
    for ri, (view, win) in enumerate(rows):
        y = n_rows - ri - 1
        if view != prev_view:
            if prev_view is not None:
                ax.axhline(y + 1, color='#555', linewidth=1.5)
            group_starts[view] = y
            prev_view = view

    # y-tick labels
    ytick_pos    = [n_rows - ri - 0.5 for ri in range(n_rows)]
    ytick_labels = [WINDOW_LABEL[win] for _, win in rows]
    ax.set_yticks(ytick_pos)
    ax.set_yticklabels(ytick_labels, fontsize=8.5)

    # view group labels on the right
    for view, y_top in group_starts.items():
        windows = ['global'] if view == 'v2dx_v3xf' else WINDOW_ORDER
        y_bottom = y_top - len(windows) + 1
        y_mid    = (y_top + y_bottom) / 2 + 0.5
        ax.text(2.08, y_mid, VIEW_LABEL[view],
                ha='left', va='center', fontsize=8.5,
                fontweight='bold', color=VIEW_COLOR[view],
                rotation=0)

    # x-axis
    ax.set_xticks([0.5, 1.5])
    ax.set_xticklabels([ENC_LABEL[e] for e in ENCODERS], fontsize=9)
    ax.xaxis.set_ticks_position('top')
    ax.xaxis.set_label_position('top')

    ax.set_xlim(0, 2)
    ax.set_ylim(0, n_rows)
    ax.set_aspect('equal')
    ax.set_title(title, fontsize=11, fontweight='bold', pad=24)

    # reference annotation for dx+xf score
    ref = scores.get(('v2dx_v3xf', 'global', 'transformer'), None)
    if ref is not None:
        ax.set_xlabel(f'Reference (dx+xf, transformer): {ref:.4f}',
                      fontsize=8, style='italic', labelpad=6)
        ax.xaxis.set_label_position('bottom')

    # colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, fraction=0.03, pad=0.16, aspect=30)
    cbar.set_label('Accuracy', fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    ax.spines[:].set_visible(False)
    ax.tick_params(length=0)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

epilepsy_scores = read_scores(
    'out_finetune/_DA_Epilepsy_256_00/final_test_metric_summary.tsv')

fig, ax = plt.subplots(figsize=(7, 9))
fig.subplots_adjust(left=0.18, right=0.72, top=0.92, bottom=0.06)

plot_heatmap(epilepsy_scores,
             'Epilepsy (SleepEEG pretrain)\nFinetune accuracy', ax)

plt.savefig('finetune_heatmap.pdf', bbox_inches='tight')
plt.savefig('finetune_heatmap.png', dpi=150, bbox_inches='tight')
print('Saved: finetune_heatmap.pdf  finetune_heatmap.png')
plt.show()

"""
visualize_results.py — Full results comparison across datasets, splits and modes.

Layout: 3 panels (Epilepsy | HAR70plus unstratified | HAR70plus stratified)
X-axis: view combination  |  Color: mode  |  Hatch: encoder type
Probes (HAR70plus stratified only) shown as single bars with no hatch.

Probe descriptions (shown as figure footnote):
  probe_raw  — Linear head G applied to mean-pooled raw view features.
               No encoder, no pretraining. Establishes a floor baseline.
  probe_pt   — Linear head G on frozen pretrained encoder projector outputs
               (zt, zd, zf). Standard linear-probe evaluation of SSL
               representation quality in the transfer-learning setting.

Usage: python visualize_results.py
"""

import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VIEW_LABEL = {
    'v2dx_v3xf':     'dx + xf',
    'v2logsig_v3xf': 'logsig + xf',
    'v2dx_v3logsig': 'dx + logsig',
}
VIEW_ORDER = list(VIEW_LABEL.keys())

MODES = ['finetune', 'freeze', 'baseline', 'probe_raw', 'probe_pt']
MODE_LABEL = {
    'finetune':  'Finetune',
    'freeze':    'Freeze',
    'baseline':  'Baseline',
    'probe_raw': 'Probe (raw)',
    'probe_pt':  'Probe (pretrained)',
}
MODE_COLOR = {
    'finetune':  '#4C72B0',
    'freeze':    '#DD8452',
    'baseline':  '#55A868',
    'probe_raw': '#C44E52',
    'probe_pt':  '#8172B3',
}
PROBE_MODES = {'probe_raw', 'probe_pt'}

ENCODERS = ['transformer', 'mlp_logsig']
ENC_HATCH = {'transformer': '', 'mlp_logsig': '///'}
ENC_LABEL = {'transformer': 'Transformer', 'mlp_logsig': 'MLP-LogSig'}

# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def parse_row(name: str):
    """
    Parse a run_name into (view_key, mode, encoder).
    Returns None for header/comment lines or unrecognised formats.
    """
    if not name or name.startswith('#') or name.startswith('run_name'):
        return None

    # --- probe rows ---
    if name.startswith('probe_raw_'):
        m = re.search(r'(v2\w+_v3\w+?)(?:_seed|$)', name)
        if m and m.group(1) in VIEW_LABEL:
            return m.group(1), 'probe_raw', 'transformer'   # no encoder; stored as transformer slot
        return None

    if name.startswith('probe_pt_'):
        m = re.search(r'(v2\w+_v3\w+?)(?:_ep|$)', name)
        if m and m.group(1) in VIEW_LABEL:
            return m.group(1), 'probe_pt', 'transformer'
        return None

    # --- standard finetune/freeze/baseline rows ---
    for mode in ('finetune', 'freeze', 'baseline'):
        if name.endswith('_' + mode):
            encoder = 'mlp_logsig' if '_mlp_logsig_' in name else 'transformer'
            m = re.search(r'(v2\w+_v3\w+?)_ep', name)
            if m and m.group(1) in VIEW_LABEL:
                return m.group(1), mode, encoder
            return None

    return None


def read_tsv_sections(path: str):
    """
    Read a summary TSV into two dicts keyed by (view_key, mode, encoder) → score.
    section_a = rows before any '####' comment line
    section_b = rows after  any '####' comment line
    """
    section_a, section_b = {}, {}
    current = section_a

    with open(path) as f:
        for line in f:
            line = line.rstrip('\n')
            if not line:
                continue
            if line.startswith('#'):
                current = section_b
                continue
            parts = line.split('\t')
            if len(parts) < 2:
                continue
            parsed = parse_row(parts[0])
            if parsed is None:
                continue
            try:
                score = float(parts[1])
            except ValueError:
                continue
            view_key, mode, encoder = parsed
            # last write wins (handles duplicates gracefully)
            current[(view_key, mode, encoder)] = score

    return section_a, section_b


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

epilepsy_data, _ = read_tsv_sections(
    'out_finetune/_DA_Epilepsy_256_00/final_test_metric_summary.tsv')

har_unstrat, har_strat = read_tsv_sections(
    'out_finetune/_DA_HAR70plus_256_00/final_test_metric_summary.tsv')

panels = [
    ('Epilepsy\n(SleepEEG pretrain)',              epilepsy_data),
    ('HAR70plus — unstratified\n(HARTH pretrain)', har_unstrat),
    ('HAR70plus — stratified\n(HARTH pretrain)',   har_strat),
]

# ---------------------------------------------------------------------------
# Bar layout helpers
# ---------------------------------------------------------------------------

def bars_for_view(data: dict, view_key: str):
    """
    Return ordered list of (mode, encoder, score_or_nan) for one view group.
    Order: [finetune-T, finetune-MLP, freeze-T, freeze-MLP,
            baseline-T, baseline-MLP, probe_raw, probe_pt]
    """
    result = []
    for mode in MODES:
        if mode in PROBE_MODES:
            score = data.get((view_key, mode, 'transformer'), float('nan'))
            result.append((mode, None, score))
        else:
            for enc in ENCODERS:
                score = data.get((view_key, mode, enc), float('nan'))
                result.append((mode, enc, score))
    return result


# Compute bar positions within a view group
def group_x_positions(n_bars, bar_w, gap):
    """Centre a group of n_bars around 0, with internal gap between sub-groups."""
    # sub-groups: (finetune×2, freeze×2, baseline×2, probe_raw×1, probe_pt×1)
    sub_widths = [2 * bar_w, 2 * bar_w, 2 * bar_w, bar_w, bar_w]
    sub_gaps = gap  # gap between sub-groups
    total = sum(sub_widths) + sub_gaps * (len(sub_widths) - 1)
    positions = []
    x = -total / 2
    for w in sub_widths:
        if w == 2 * bar_w:
            positions.extend([x + bar_w / 2, x + bar_w * 1.5])
        else:
            positions.append(x + bar_w / 2)
        x += w + sub_gaps
    return positions, total


BAR_W = 0.10
GROUP_SEP = 0.06   # gap between sub-groups inside a view cluster
VIEW_SEP = 1.0     # centre-to-centre distance between view clusters

rel_pos, group_width = group_x_positions(8, BAR_W, GROUP_SEP)

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=False)
fig.suptitle('Classification accuracy by mode, view combination and encoder',
             fontsize=13, fontweight='bold', y=1.02)

for ax, (title, data) in zip(axes, panels):
    x_centres = np.arange(len(VIEW_ORDER)) * VIEW_SEP

    for vi, view_key in enumerate(VIEW_ORDER):
        bars_info = bars_for_view(data, view_key)
        for bi, (mode, enc, score) in enumerate(bars_info):
            xpos = x_centres[vi] + rel_pos[bi]
            hatch = ENC_HATCH.get(enc, '') if enc is not None else ''
            color = MODE_COLOR[mode]
            ax.bar(xpos, score if not np.isnan(score) else 0,
                   width=BAR_W - 0.01,
                   color=color,
                   hatch=hatch,
                   edgecolor='white' if hatch == '' else 'black',
                   linewidth=0.5,
                   zorder=3)
            if not np.isnan(score):
                ax.text(xpos, score + 0.003,
                        f'{score:.3f}',
                        ha='center', va='bottom',
                        fontsize=5.5, rotation=90)

    ax.set_title(title, fontsize=10, pad=8)
    ax.set_xticks(x_centres)
    ax.set_xticklabels([VIEW_LABEL[v] for v in VIEW_ORDER], fontsize=9)
    ax.set_ylabel('Accuracy', fontsize=9)
    ax.set_ylim(0.0, 1.15)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.2f}'))
    ax.grid(axis='y', linestyle='--', alpha=0.4, zorder=0)
    ax.spines[['top', 'right']].set_visible(False)
    ax.set_xlim(-VIEW_SEP * 0.55, (len(VIEW_ORDER) - 1) * VIEW_SEP + VIEW_SEP * 0.55)

# ---------------------------------------------------------------------------
# Legend
# ---------------------------------------------------------------------------

mode_patches = [
    mpatches.Patch(color=MODE_COLOR[m], label=MODE_LABEL[m])
    for m in MODES
]
enc_patches = [
    mpatches.Patch(facecolor='grey', hatch=ENC_HATCH[e],
                   edgecolor='black', label=ENC_LABEL[e])
    for e in ENCODERS
]

leg1 = fig.legend(handles=mode_patches, loc='lower center',
                  bbox_to_anchor=(0.35, -0.06), ncol=5,
                  fontsize=8.5, frameon=False,
                  title='Mode', title_fontsize=9)
leg2 = fig.legend(handles=enc_patches, loc='lower center',
                  bbox_to_anchor=(0.80, -0.06), ncol=2,
                  fontsize=8.5, frameon=False,
                  title='Encoder (hatch)', title_fontsize=9)
fig.add_artist(leg1)

# ---------------------------------------------------------------------------
# Probe description footnote
# ---------------------------------------------------------------------------

probe_note = (
    'Probe (raw): linear head G on mean-pooled raw views — no encoder, no pretraining. Floor baseline.\n'
    'Probe (pretrained): linear head G on frozen pretrained encoder outputs (zt, zd, zf). '
    'Linear-probing of SSL representation quality in the transfer-learning setting.'
)
fig.text(0.5, -0.12, probe_note, ha='center', va='top',
         fontsize=8, style='italic',
         bbox=dict(boxstyle='round,pad=0.4', facecolor='#f5f5f5', edgecolor='#cccccc'))

plt.tight_layout()
plt.savefig('finetune_results.pdf', bbox_inches='tight')
plt.savefig('finetune_results.png', dpi=150, bbox_inches='tight')
print('Saved: finetune_results.pdf  finetune_results.png')
plt.show()

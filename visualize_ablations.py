"""
visualize_ablations.py — Ablation analysis for Sections G, H, I.

Section G: multi-smooth Tukey windows (msp2=0.25+0.5, msp3=0.25+0.5+0.75 vs sp=0.5 baseline)
Section H: log-signature depth (d2 baseline, d3, d4) — Epilepsy only (HAR70+ pending)
Section I: Tukey taper ratio (sp 0.1 0.25 0.5 0.75 0.9) — Epilepsy complete, HAR70+ partial

Layout: 3 columns (G/H/I) × 2 rows (Epilepsy / HAR70+)
Each panel: mean ± 95% CI across 10 seeds, one line per view combination.

Usage: python visualize_ablations.py
"""

import re
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

matplotlib.rcParams.update({
    'font.size': 10,
    'axes.titlesize': 11,
    'axes.labelsize': 10,
    'legend.fontsize': 9,
    'hatch.linewidth': 1.5,
})

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DATASETS = [
    ('_DA_Epilepsy_256_00',  'SleepEEG → Epilepsy',
     'out_finetune/_DA_Epilepsy_256_00/final_test_metric_summary.tsv'),
    ('_DA_HAR70plus_256_00', 'HARTH → HAR70+',
     'out_finetune/_DA_HAR70plus_256_00/final_test_metric_summary.tsv'),
]

VIEW_KEYS  = ['v2logsig_v3xf', 'v2dx_v3logsig', 'v2logsig_nview']
VIEW_LABEL = {'v2logsig_v3xf': 'logsig+xf', 'v2dx_v3logsig': 'dx+logsig', 'v2logsig_nview': 'logsig (2-view)'}
VIEW_COLOR = {'v2logsig_v3xf': '#DD8452', 'v2dx_v3logsig': '#55A868', 'v2logsig_nview': '#C44E52'}
VIEW_MARKER = {'v2logsig_v3xf': 'o', 'v2dx_v3logsig': 's', 'v2logsig_nview': '^'}

# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def _view_key(name):
    for v in VIEW_KEYS:
        if v in name:
            return v
    return None

def _ablation_category(name):
    """Return (section, key) for ablation rows, or None for unrelated rows."""
    m = re.search(r'_msp(\d+)_ilbilinear', name)
    if m:
        return ('G', f'msp{m.group(1)}')
    m = re.search(r'_d([34])_ilbilinear', name)
    if m:
        return ('H', f'd{m.group(1)}')
    m = re.search(r'_sp([\d.]+)_ilbilinear', name)
    if m:
        return ('I', f'sp{m.group(1)}')
    # baseline: mlp_logsig tukey128 bilinear, no msp/sp/depth modifier
    if re.search(r'mlp_logsig_tukey128_ilbilinear', name) and \
       not re.search(r'(msp|sp[\d.]+|_d[34])', name):
        return ('base', 'sp0.5')
    return None


def load_ablations(tsv_path):
    """Return dict: section → view_key → label → list[score]."""
    df = pd.read_csv(tsv_path, sep='\t')
    data = {}  # section → view → label → [scores]
    for _, row in df.iterrows():
        name  = row['run_name']
        score = float(row['final_test_score'])
        view  = _view_key(name)
        ab    = _ablation_category(name)
        if view is None or ab is None:
            continue
        sec, lbl = ab
        data.setdefault(sec, {}).setdefault(view, {}).setdefault(lbl, []).append(score)
    return data


# G x-axis: baseline + msp2 + msp3
G_LABELS     = ['sp0.5',   'msp2',  'msp3']
G_X          = [0.5,        1.0,     1.5]
G_TICK_LABEL = ['sp=0.5\n(baseline)', 'msp(0.25,0.5)', 'msp(0.25,0.5,0.75)']

# H x-axis: d2 baseline + d3 + d4
H_LABELS     = ['sp0.5',  'd3',    'd4']
H_X          = [2,         3,       4]
H_TICK_LABEL = ['d=2\n(baseline)', 'd=3', 'd=4']

# I x-axis: all smooth params including baseline sp=0.5
I_LABELS     = ['sp0.1', 'sp0.25', 'sp0.5', 'sp0.75', 'sp0.9']
I_X          = [0.1, 0.25, 0.5, 0.75, 0.9]
I_TICK_LABEL = ['0.1', '0.25', '0.5\n(base)', '0.75', '0.9']


def _ci95(vals):
    """Return (mean, half-CI) using t-distribution for small samples."""
    from scipy import stats as st
    n = len(vals)
    if n == 0:
        return np.nan, np.nan
    if n == 1:
        return vals[0], 0.0
    m = np.mean(vals)
    se = np.std(vals, ddof=1) / np.sqrt(n)
    ci = st.t.ppf(0.975, df=n-1) * se
    return m, ci


def _plot_panel(ax, data, section, x_vals, labels, tick_labels,
                ylabel=True, title='', annotate_n=True):
    """Plot one ablation panel."""
    sec_data = data.get(section, {})
    base_data = data.get('base', {})

    has_any = False
    for vk in VIEW_KEYS:
        vd = sec_data.get(vk, {})
        bd = base_data.get(vk, {})
        ys, errs = [], []
        xs_used = []

        for x, lbl in zip(x_vals, labels):
            if lbl == 'sp0.5':          # pull baseline from 'base' section
                vals = bd.get('sp0.5', [])
            else:
                vals = vd.get(lbl, [])
            if vals:
                m, ci = _ci95(vals)
                ys.append(m)
                errs.append(ci)
                xs_used.append(x)
            else:
                ys.append(np.nan)
                errs.append(np.nan)
                xs_used.append(x)

        if all(np.isnan(ys)):
            continue

        has_any = True
        ax.errorbar(xs_used, ys, yerr=errs,
                    color=VIEW_COLOR[vk], marker=VIEW_MARKER[vk],
                    label=VIEW_LABEL[vk],
                    linewidth=1.8, markersize=6, capsize=3,
                    linestyle='-')

    if not has_any:
        ax.text(0.5, 0.5, 'data pending', transform=ax.transAxes,
                ha='center', va='center', color='grey', fontsize=11)

    ax.set_xticks(x_vals)
    ax.set_xticklabels(tick_labels, fontsize=8.5)
    ax.set_title(title, fontsize=10.5, pad=4)
    if ylabel:
        ax.set_ylabel('Test accuracy', fontsize=9)
    ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%.3f'))
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Annotate seed count on first non-nan point
    if annotate_n and has_any:
        for vk in VIEW_KEYS:
            vd = sec_data.get(vk, {})
            bd = base_data.get(vk, {})
            for x, lbl in zip(x_vals, labels):
                vals = bd.get('sp0.5', []) if lbl == 'sp0.5' else vd.get(lbl, [])
                if vals:
                    n = len(vals)
                    ax.annotate(f'n={n}', xy=(x_vals[0], np.nanmean(
                        [_ci95(bd.get('sp0.5', []) if l == 'sp0.5' else vd.get(l, []) or [np.nan])[0]
                         for l in labels])), fontsize=0)  # invisible, just a placeholder
                    break
            break


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

fig, axes = plt.subplots(2, 3, figsize=(14, 8.5),
                         gridspec_kw={'hspace': 0.45, 'wspace': 0.30})

col_titles = [
    'Section G — Multi-smooth Tukey',
    'Section H — Log-signature depth',
    'Section I — Tukey taper ratio (α)',
]
row_labels = ['SleepEEG → Epilepsy', 'HARTH → HAR70+']

for row_idx, (ds_key, ds_label, tsv_path) in enumerate(DATASETS):
    try:
        data = load_ablations(tsv_path)
    except FileNotFoundError:
        for col in range(3):
            axes[row_idx, col].text(0.5, 0.5, 'no data', transform=axes[row_idx, col].transAxes,
                                     ha='center', va='center', color='grey')
        continue

    ax_G = axes[row_idx, 0]
    ax_H = axes[row_idx, 1]
    ax_I = axes[row_idx, 2]

    is_left = True

    _plot_panel(ax_G, data, 'G',
                G_X, G_LABELS, G_TICK_LABEL,
                ylabel=True,
                title=col_titles[0] if row_idx == 0 else '')

    _plot_panel(ax_H, data, 'H',
                H_X, H_LABELS, H_TICK_LABEL,
                ylabel=False,
                title=col_titles[1] if row_idx == 0 else '')

    _plot_panel(ax_I, data, 'I',
                I_X, I_LABELS, I_TICK_LABEL,
                ylabel=False,
                title=col_titles[2] if row_idx == 0 else '')

    ax_I.set_xlabel('Tukey taper ratio α', fontsize=9)

    # Row label
    axes[row_idx, 0].set_ylabel(f'{ds_label}\n\nTest accuracy', fontsize=9)

# Shared legend (bottom)
handles = [mpatches.Patch(color=VIEW_COLOR[v], label=VIEW_LABEL[v]) for v in VIEW_KEYS]
fig.legend(handles=handles, loc='lower center', ncol=3, frameon=False,
           fontsize=10, bbox_to_anchor=(0.5, -0.01))

fig.suptitle('Ablation Analysis: Multi-smooth (G), Depth (H), Taper α (I)\n'
             'MLP-LogSig encoder · bilinear interaction · tukey128 · ep=2',
             fontsize=12, y=1.01)

plt.tight_layout()
out_png = 'ablation_analysis.png'
out_pdf = 'ablation_analysis.pdf'
plt.savefig(out_png, dpi=150, bbox_inches='tight')
plt.savefig(out_pdf, bbox_inches='tight')
print(f'Saved: {out_png}  {out_pdf}')

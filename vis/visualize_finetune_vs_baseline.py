"""
visualize_finetune_vs_baseline.py — Finetune vs. random-init baseline, capture24 pretrain.

Reuses the categorical-scatter layout from visualize_capture24.py (view-combo
groups on the x-axis, encoder slots, window/adjustment markers) and adds a
dumbbell at every point that has baseline data: a grey hollow marker for the
random-init baseline run, joined to the coloured finetune marker by a thin
connector, so the pretraining gain is visible directly at each point.

Baseline (`_baseline`) runs are random-init, so their score doesn't depend on
which dataset they were pretrained "alongside" (the run name still carries a
pt-tag for bookkeeping) — baseline scores for a given (view, window, encoder,
channel_adapt) are pooled across all pt-tags found in the target's TSV.
Baseline coverage is much sparser than finetune, so most points simply have no
dumbbell.

Usage:
  python visualize_finetune_vs_baseline.py
  python visualize_finetune_vs_baseline.py --datasets WISDM HAR70plus
  python visualize_finetune_vs_baseline.py --metric f1_macro
"""

import argparse
import os
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.transforms import blended_transform_factory
import matplotlib.pyplot as plt

import visualize_capture24 as vc

BASELINE_COLOR = '#999999'
BASELINE_MARKER_SIZE = 4.5


def parse_baseline_row(name: str):
    """Return (view_key, window, encoder, channel_adapt) or None.
    Pools across pt-tags, since baseline is random-init and pretrain-agnostic."""
    if not name or not name.endswith('_baseline'):
        return None
    return vc.parse_run_row(name)


def plot_dataset(ax, ft_scores, bl_scores, simmtm_vals, title, metric_label, baseline_mean=None):
    """Draw one panel: capture24_results layout, with a baseline dumbbell added
    at every point where baseline data exists."""

    for si, (lo, hi) in enumerate(vc._slot_tiles):
        if si % 2 == 1:
            ax.axvspan(lo, hi, color='#f0f0f0', zorder=0, linewidth=0)

    y_min_seen = 1.0
    for vi, view_key in enumerate(vc.VIEW_ORDER):
        xc = vc.view_centres[vi]
        specs, _, _ = vc._VIEW_SPECS[view_key]

        for win, enc, ca, rel_x in specs:
            x = xc + rel_x

            if enc == 'simmtm':
                if not simmtm_vals:
                    continue
                mu, ci = vc._mean_ci(simmtm_vals)
                yerr = None if ci is None else [[min(ci, mu)], [min(ci, 1.0 - mu)]]
                ax.errorbar(x, mu, yerr=yerr,
                            fmt=vc.SIMMTM_MARKER, markersize=vc.SIMMTM_SIZE,
                            markerfacecolor=vc.SIMMTM_COLOR, markeredgecolor=vc.SIMMTM_COLOR,
                            ecolor=vc.SIMMTM_COLOR, elinewidth=1.1, capsize=2, zorder=5)
                continue

            ft_vals = ft_scores.get((view_key, win, enc, ca), [])
            ft_mu, ft_ci = vc._mean_ci(ft_vals)
            if ft_mu is None:
                continue
            method = vc.MAJOR_METHOD[win]
            adj    = vc.ADJUSTMENT[win]

            bl_vals = bl_scores.get((view_key, win, enc, ca), [])
            bl_mu, bl_ci = vc._mean_ci(bl_vals)
            if bl_mu is not None:
                ax.plot([x, x], [bl_mu, ft_mu], color=BASELINE_COLOR,
                        linewidth=1.0, zorder=3)
                bl_yerr = None if bl_ci is None else [[min(bl_ci, bl_mu)], [min(bl_ci, 1.0 - bl_mu)]]
                ax.errorbar(x, bl_mu, yerr=bl_yerr,
                            fmt='o', markersize=BASELINE_MARKER_SIZE,
                            markerfacecolor='none', markeredgecolor=BASELINE_COLOR,
                            markeredgewidth=1.1,
                            ecolor=BASELINE_COLOR, elinewidth=0.9, capsize=1.5, zorder=4)
                y_min_seen = min(y_min_seen, bl_mu - (bl_ci or 0.0))

            ft_yerr = None if ft_ci is None else [[min(ft_ci, ft_mu)], [min(ft_ci, 1.0 - ft_mu)]]
            ax.errorbar(x, ft_mu, yerr=ft_yerr,
                        fmt=vc.METHOD_MARKER[method],
                        markersize=vc.MARKER_SIZE,
                        markerfacecolor=vc.ADJUSTMENT_COLOR[adj],
                        markeredgecolor='#333',
                        markeredgewidth=0.6,
                        ecolor='#333',
                        elinewidth=1.1,
                        capsize=2,
                        zorder=5)

            overlay = vc.CHADAPT_OVERLAY[ca]
            if overlay is not None:
                ax.plot(x, ft_mu, marker=overlay, color='black',
                        markersize=4.2, markeredgewidth=1.3, zorder=6)

        if vi < len(vc.VIEW_ORDER) - 1:
            ax.axvline(vc._group_bounds[vi + 1], color='#999999',
                       linewidth=1.0, linestyle='-', zorder=1)

    for sep_x in vc._slot_separators:
        ax.axvline(sep_x, color='#cccccc', linewidth=0.6, linestyle=':', zorder=1)

    if baseline_mean is not None:
        ax.axhline(baseline_mean, color='black', linestyle=':', linewidth=1.1, zorder=4)
        ax.text(vc.view_centres[-1] + vc.VIEW_SEP * 0.45, baseline_mean + 0.004,
                f'ref {baseline_mean:.3f}', fontsize=6.5, va='bottom', ha='right', color='#222')

    ax.set_xticks(vc._slot_ticks)
    ax.set_xticklabels(vc._slot_labels, fontsize=6.3, rotation=50, ha='right')
    ax.tick_params(axis='x', pad=2)

    trans = blended_transform_factory(ax.transData, ax.transAxes)
    bracket_y, label_y = -0.42, -0.47
    for vk, gxc, ghalf in vc._group_ranges:
        ax.plot([gxc - ghalf, gxc + ghalf], [bracket_y, bracket_y],
                transform=trans, color='#555', linewidth=0.9,
                clip_on=False, zorder=2)
        ax.annotate(vc.VIEW_LABEL[vk], xy=(gxc, label_y), xycoords=trans,
                    ha='center', va='top', fontsize=7, color='#222',
                    annotation_clip=False)

    ax.set_xlim(vc._group_bounds[0] - 0.1, vc._group_bounds[-1] + 0.1)
    ax.set_ylabel(metric_label, fontsize=9)
    ax.set_title(title, fontsize=10, fontweight='bold')
    ax.set_ylim(max(0.0, min(0.7, y_min_seen) - 0.03), 1.05)
    ax.yaxis.grid(True, linewidth=0.4, alpha=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.spines[['top', 'right']].set_visible(False)

    cx0 = vc.view_centres[0]
    all_n = [len(ft_scores.get((vc.VIEW_ORDER[0], 'global', e, 'none'), []))
             for e in ['transformer', 'transformer_bilinear']]
    n = max(all_n) if all_n else 0
    if n > 0:
        ax.text(cx0, 0.02, f'n={n}', ha='center', va='bottom',
                fontsize=7, color='gray', transform=trans)


def build_legend(fig):
    method_handles = [
        Line2D([0], [0], marker=vc.METHOD_MARKER[m], linestyle='none',
               markerfacecolor='#888', markeredgecolor='#333', markersize=8,
               label=vc.METHOD_LABEL[m])
        for m in vc.METHOD_ORDER
    ] + [
        Line2D([0], [0], marker=vc.SIMMTM_MARKER, linestyle='none',
               markerfacecolor=vc.SIMMTM_COLOR, markeredgecolor=vc.SIMMTM_COLOR,
               markersize=10, label='SimMTM'),
    ]
    adjustment_handles = [
        Line2D([0], [0], marker='o', linestyle='none',
               markerfacecolor=vc.ADJUSTMENT_COLOR[a], markeredgecolor='#333', markersize=8,
               label=vc.ADJUSTMENT_LABEL[a])
        for a in vc.ADJUSTMENT_ORDER
    ]
    baseline_handles = [
        Line2D([0], [0], marker='o', linestyle='-', color=BASELINE_COLOR,
               markerfacecolor='none', markeredgecolor=BASELINE_COLOR, markersize=7,
               label='baseline (random init)'),
    ]

    leg1 = fig.legend(handles=method_handles,
                      loc='lower center', bbox_to_anchor=(0.16, -0.04),
                      ncol=4, fontsize=8, frameon=False,
                      title='Major method / comparison', title_fontsize=8.5)
    leg2 = fig.legend(handles=adjustment_handles,
                      loc='lower center', bbox_to_anchor=(0.58, -0.04),
                      ncol=5, fontsize=8, frameon=False,
                      title='Adjustment', title_fontsize=8.5)
    fig.legend(handles=baseline_handles,
               loc='lower center', bbox_to_anchor=(0.92, -0.04),
               ncol=1, fontsize=8, frameon=False,
               title='Dumbbell', title_fontsize=8.5)
    fig.add_artist(leg1)
    fig.add_artist(leg2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets', nargs='+',
                        default=[d[0] for d in vc.ALL_DATASETS],
                        choices=[d[0] for d in vc.ALL_DATASETS],
                        help='Which datasets to plot (default: all available)')
    parser.add_argument('--metric', default='accuracy',
                        choices=['accuracy', 'f1_macro'],
                        help='Metric column to visualise')
    parser.add_argument('--out', default='plots/capture24_finetune_vs_baseline.pdf',
                        help='Output file (pdf/png/svg)')
    args = parser.parse_args()

    _simmtm_scores = vc.read_simmtm_tsv(vc.SIMMTM_TSV, metric=args.metric)

    requested = {d[0] for d in vc.ALL_DATASETS if d[0] in args.datasets}
    panels = []
    for (short, data_tag, label) in vc.ALL_DATASETS:
        if short not in requested:
            continue
        tsv = f'out_finetune/{data_tag}/final_test_metric_summary.tsv'
        ft_scores = vc.read_tsv(tsv)
        bl_scores = vc.read_tsv(tsv, parser=parse_baseline_row)
        if not ft_scores:
            print(f'  Skipping {short}: no data in {tsv}')
            continue
        simmtm_vals = _simmtm_scores.get(('capture24', short), [])
        panels.append((label, ft_scores, bl_scores, simmtm_vals))

    if not panels:
        print('No data found for any requested dataset.')
        return

    metric_label = 'Test accuracy' if args.metric == 'accuracy' else 'Test F1'

    ncols = min(2, len(panels))
    nrows = (len(panels) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(13 * ncols, 6.5 * nrows),
                             squeeze=False)

    for idx, (label, ft_scores, bl_scores, simmtm_vals) in enumerate(panels):
        ax = axes[idx // ncols][idx % ncols]
        baseline_vals = ft_scores.get(('v2dx_v3xf', 'global', 'transformer', 'none'), [])
        if not baseline_vals:
            baseline_vals = ft_scores.get(('v2dx_v3xf', 'global', 'transformer_bilinear', 'none'), [])
        baseline_mean, _ = vc._mean_ci(baseline_vals)
        plot_dataset(ax, ft_scores, bl_scores, simmtm_vals,
                     f'capture24 → {label}', metric_label, baseline_mean)

    for idx in range(len(panels), nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    build_legend(fig)
    fig.suptitle(f'Multi-View Contrastive Learning — finetune vs. baseline, capture24 pretrain ({metric_label})',
                 fontsize=13, fontweight='bold', y=1.01)
    fig.subplots_adjust(hspace=1.1, wspace=0.25, top=0.96, bottom=0.12)

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    plt.savefig(args.out, bbox_inches='tight', dpi=150)
    base, _ = os.path.splitext(args.out)
    if not args.out.endswith('.png'):
        plt.savefig(base + '.png', bbox_inches='tight', dpi=150)
    print(f'Saved → {args.out}')
    print(f'Datasets plotted: {[p[0] for p in panels]}')


if __name__ == '__main__':
    main()

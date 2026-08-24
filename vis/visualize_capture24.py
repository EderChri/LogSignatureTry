"""
visualize_capture24.py — Test accuracy for capture24 pretrain → all finetune targets.

Usage:
  python visualize_capture24.py                 # all available datasets
  python visualize_capture24.py --datasets WISDM HAR70plus
  python visualize_capture24.py --metric f1_macro

Layout: one subplot per dataset, same categorical-scatter structure as
visualize_results.py — view-combo groups on the x-axis, ordered within each
group by model type; each point is a (major method shape, adjustment colour)
marker with a 95% CI whisker; channel_adapt (drop/pca/copy) is an overlay mark
on top of its base point (win64_norm_ll, norm_win64, norm_win64_lag, and the
v2dx_v3xf baseline), since this file is exactly where the capture24-pretrained
channel-adapt trials (experiments_channels.json) live.
Missing datasets are skipped without error.
"""

import argparse
import os
import re
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.transforms import blended_transform_factory
from scipy import stats

# ---------------------------------------------------------------------------
# Datasets (in display order)
# ---------------------------------------------------------------------------

ALL_DATASETS = [
    ('WISDM',       '_DA_WISDM_256_00',       'WISDM'),
    ('WISDM2',      '_DA_WISDM2_256_00',       'WISDM2'),
    ('HAR70plus',   '_DA_HAR70plus_256_00',    'HAR70plus'),
    ('USC_HAD',     '_DA_USC_HAD_256_00',      'USC-HAD'),
    ('Opportunity', '_DA_Opportunity_256_00',  'Opportunity (113ch→32 PCA)'),
    ('Skoda',       '_DA_Skoda_256_00',        'Skoda (60ch→4 PCA)'),
]

PRETRAIN_TAG = '_DA_capture24_256_00'   # full capture24 only (not mini)

# ---------------------------------------------------------------------------
# View / window / encoder constants (shared vocabulary with visualize_results.py)
# ---------------------------------------------------------------------------

VIEW_ORDER = ['v2dx_v3xf', 'v2logsig_v3xf', 'v2dx_v3logsig', 'v2logsig_nview']
VIEW_LABEL = {
    'v2dx_v3xf':      'dx + xf\n(baseline)',
    'v2logsig_v3xf':  'logsig + xf',
    'v2dx_v3logsig':  'dx + logsig',
    'v2logsig_nview': 'logsig\n(2-view)',
}

# Comment out any entry here to hide it from all plots.
DISABLED_WINDOWS = {
    # 'global',
    # 'win64',
    # 'win64_sl1',
    # 'win64_ll',
    # 'win64_rp',
    # 'win64_ll_rp',
    # 'win64_norm_ll',
    # 'norm_win64',
    # 'norm_win64_lag',
    # 'win64_lsn0.1',
    'win128',
    'win128_s7',
    # 'tukey64',
    # 'tukey128',
    # 'tukey128_s7',
}

_ALL_WINDOW_ORDER = ['global', 'win64', 'win64_sl1', 'win64_ll', 'win64_rp', 'win64_ll_rp',
                     'win64_norm_ll', 'win64_lsn0.1', 'norm_win64', 'norm_win64_lag',
                     'win128', 'win128_s7', 'tukey64', 'tukey128', 'tukey128_d1',
                     'tukey128_gt', 'tukey128_plast', 'tukey128_gt_plast',
                     'tukey128_ll', 'tukey128_rp', 'tukey128_ll_rp', 'tukey128_s7']
WINDOW_ORDER = [w for w in _ALL_WINDOW_ORDER if w not in DISABLED_WINDOWS]

# ---------------------------------------------------------------------------
# Figure vocabulary: major method (marker shape) x adjustment (marker colour).
# Every window key above factors into exactly one (method, adjustment) pair.
# ---------------------------------------------------------------------------

MAJOR_METHOD = {
    'global':         'global',
    'win64':          'win64',
    'win64_sl1':      'win64',
    'win64_ll':       'win64',
    'win64_rp':       'win64',
    'win64_ll_rp':    'win64',
    'win64_norm_ll':  'win64',
    'win64_lsn0.1':   'win64',
    'norm_win64':     'norm_win64',
    'norm_win64_lag': 'norm_win64',
    'win128':         'win128',
    'win128_s7':      'win128',
    'tukey64':        'tukey64',
    'tukey128':       'tukey128',
    'tukey128_d1':       'tukey128',
    'tukey128_gt':       'tukey128',
    'tukey128_plast':    'tukey128',
    'tukey128_gt_plast': 'tukey128',
    'tukey128_ll':    'tukey128',
    'tukey128_rp':    'tukey128',
    'tukey128_ll_rp': 'tukey128',
    'tukey128_s7':    'tukey128',
}
ADJUSTMENT = {
    'global':         'none',
    'win64':          'none',
    'win64_sl1':      'no_lv1',
    'win64_ll':       'lead_lag',
    'win64_rp':       'reduced_pen',
    'win64_ll_rp':    'lead_lag_rp',
    'win64_norm_ll':  'norm_leadlag',
    'win64_lsn0.1':   'noise',
    'norm_win64':     'none',
    'norm_win64_lag': 'lag',
    'win128':         'none',
    'win128_s7':      'stride7',
    'tukey64':        'none',
    'tukey128':       'none',
    'tukey128_d1':       'depth1',
    'tukey128_gt':       'global_time',
    'tukey128_plast':    'pool_last',
    'tukey128_gt_plast': 'gt_pool_last',
    'tukey128_ll':    'lead_lag',
    'tukey128_rp':    'reduced_pen',
    'tukey128_ll_rp': 'lead_lag_rp',
    'tukey128_s7':    'stride7',
}

METHOD_ORDER = ['global', 'win64', 'norm_win64', 'win128', 'tukey64', 'tukey128']
METHOD_LABEL = {
    'global':     'global',
    'win64':      'win 64',
    'norm_win64': 'norm win 64',
    'win128':     'win 128',
    'tukey64':    'tukey 64',
    'tukey128':   'tukey 128',
}
METHOD_MARKER = {
    'global':     'o',
    'win64':      's',
    'norm_win64': '^',
    'win128':     'D',
    'tukey64':    'v',
    'tukey128':   'P',
}

ADJUSTMENT_ORDER = ['none', 'no_lv1', 'lead_lag', 'reduced_pen', 'lead_lag_rp',
                    'norm_leadlag', 'noise', 'lag', 'stride7',
                    'depth1', 'global_time', 'pool_last', 'gt_pool_last']
ADJUSTMENT_LABEL = {
    'none':         'none',
    'no_lv1':       'no lv1',
    'lead_lag':     'lead-lag',
    'reduced_pen':  'red. pen.',
    'lead_lag_rp':  'lead-lag + red. pen.',
    'norm_leadlag': 'norm + lead-lag',
    'noise':        'noise',
    'lag':          'lag',
    'stride7':      'stride 7',
    'depth1':       'depth 1',
    'global_time':  'global time',
    'pool_last':    'pool: last',
    'gt_pool_last': 'global time + pool: last',
}
_adj_palette = matplotlib.colormaps['tab20'].colors
ADJUSTMENT_COLOR = {adj: _adj_palette[i % len(_adj_palette)] for i, adj in enumerate(ADJUSTMENT_ORDER)}

# channel_adapt: overlay drawn on top of the base (method, adjustment) marker, so the
# channel-adapt family still reads as "the same point, with a mark on it".
# capture24 pretrain is the source of the channel-adapt trials, so this is the file
# where drop/pca/copy actually show up (see _CHADAPT_WINDOWS/_CHADAPT_MODEL_TYPES).
CHADAPT_ORDER = ['none', 'drop', 'pca', 'copy']
CHADAPT_OVERLAY = {'none': None, 'drop': '.', 'pca': 'x', 'copy': '+'}
CHADAPT_LABEL = {'none': 'none (no mark)', 'drop': 'drop (• overlay)',
                 'pca': 'pca (× overlay)', 'copy': 'copy (+ overlay)'}

ENCODERS = ['transformer', 'transformer_plast', 'mlp_logsig',
            'mlp_logsig_bilinear', 'transformer_bilinear']
ENC_AXIS_LABEL = {
    'transformer':          'mean pool',
    'transformer_plast':    'last pool',
    'mlp_logsig':           'MLP-LogSig',
    'mlp_logsig_bilinear':  'MLP-LogSig+bil',
    'transformer_bilinear': 'Transformer+bil',
    'simmtm':               'SimMTM',
}

SIMMTM_TSV    = 'SimMTM/SimMTM_Classification/results/simmtm_results.tsv'
SIMMTM_MARKER = '*'
SIMMTM_COLOR  = '#222222'

# ---------------------------------------------------------------------------
# Point geometry (mirrors visualize_results.py)
# ---------------------------------------------------------------------------

POINT_PITCH          = 0.080   # centre-to-centre spacing between adjacent points (same method)
BIL_POINT_PITCH      = 0.108   # wider spacing within mlp_logsig_bilinear
COMPACT_POINT_PITCH  = 0.018   # tight spacing for transformer (mean-pool) slot — halves its tile
COMPACT_METHOD_GAP   = 0.050   # compressed inter-method gap within the transformer slot
METHOD_GAP      = 0.100   # extra gap injected when the major method changes
SLOT_GAP        = 0.25    # gap between model-type slots within a view group
CHADAPT_PITCH   = 0.044   # tight centre-to-centre spacing for channel_adapt sub-positions
MARKER_SIZE     = 6.0
SIMMTM_SIZE     = MARKER_SIZE * 0.8

# win64, win128, tukey64/128, norm_win64(+lag) (+ win64_norm_ll, win64_lsn0.1) have bilinear data.
_BIL_WIN_SET = {'win64', 'win64_sl1', 'win64_ll', 'win64_rp', 'win64_ll_rp',
                'win64_norm_ll', 'win64_lsn0.1', 'norm_win64', 'norm_win64_lag',
                'win128', 'win128_s7', 'tukey64', 'tukey128', 'tukey128_d1',
                'tukey128_gt', 'tukey128_plast', 'tukey128_gt_plast',
                'tukey128_ll', 'tukey128_rp', 'tukey128_ll_rp', 'tukey128_s7'}

# windows/model-types where capture24 channel-adapt trials (drop/pca/copy) exist,
# as an overlay mark on top of the base point.
_CHADAPT_WINDOWS = {'win64_norm_ll', 'norm_win64', 'norm_win64_lag',
                     'tukey128', 'tukey128_ll', 'tukey128_rp', 'tukey128_ll_rp'}
_CHADAPT_MODEL_TYPES = {'mlp_logsig_bilinear', 'transformer_bilinear'}


def _model_types_for_window(win):
    if win == 'global':
        return ['transformer', 'transformer_plast', 'mlp_logsig',
                 'mlp_logsig_bilinear', 'transformer_bilinear']
    elif win == 'norm_win64_lag':
        return ['transformer', 'mlp_logsig', 'mlp_logsig_bilinear', 'transformer_bilinear']
    elif win in _BIL_WIN_SET:
        return ['transformer', 'mlp_logsig_bilinear', 'transformer_bilinear']
    else:
        return ['transformer', 'mlp_logsig']


def _slot_points(model_type, windows):
    """Lay out one x-position per (window, channel_adapt) item for a single
    model-type slot. Returns ([(window, channel_adapt, rel_x), ...], slot_width)
    with rel_x centred on the slot (0 = slot centre)."""
    if model_type == 'mlp_logsig_bilinear':
        pitch, meth_gap = BIL_POINT_PITCH, METHOD_GAP
    elif model_type == 'transformer':
        pitch, meth_gap = COMPACT_POINT_PITCH, COMPACT_METHOD_GAP
    else:
        pitch, meth_gap = POINT_PITCH, METHOD_GAP

    items = []
    for w in windows:
        if w in _CHADAPT_WINDOWS and model_type in _CHADAPT_MODEL_TYPES:
            items.extend((w, ca) for ca in CHADAPT_ORDER)
        else:
            items.append((w, 'none'))

    xs, x = [], 0.0
    for i, (w, _ca) in enumerate(items):
        if i > 0:
            prev_w = items[i - 1][0]
            if w == prev_w:
                x += CHADAPT_PITCH
            elif MAJOR_METHOD[w] == MAJOR_METHOD[prev_w]:
                x += pitch
            else:
                x += meth_gap
        xs.append(x)

    width = xs[-1] if xs else 0.0
    centre = width / 2
    return [(w, ca, xv - centre) for (w, ca), xv in zip(items, xs)], width


def _view_point_specs(view_key):
    """Return (specs, slot_centres, total_width) for one view group.
    specs        = [(window, model_type, channel_adapt, rel_x), ...] rel. to view centre
    slot_centres = [(model_type, rel_x_centre, half_width), ...]
    """
    if view_key == 'v2dx_v3xf':
        per_model = [('transformer', [('global', 'none', -CHADAPT_PITCH / 2),
                                       ('global', 'pca', CHADAPT_PITCH / 2)], CHADAPT_PITCH),
                     ('transformer_bilinear', [('global', 'none', 0.0)], 0.0),
                     ('simmtm', [('simmtm', 'none', 0.0)], 0.0)]
    else:
        per_model_windows = {enc: [] for enc in ENCODERS}
        for win in WINDOW_ORDER:
            for enc in _model_types_for_window(win):
                per_model_windows[enc].append(win)
        per_model = []
        for enc in ENCODERS:
            windows = per_model_windows[enc]
            if not windows:
                continue
            pts, width = _slot_points(enc, windows)
            per_model.append((enc, pts, width))

    n_gaps = max(len(per_model) - 1, 0)
    total = sum(w for _, _, w in per_model) + SLOT_GAP * n_gaps
    specs, slot_centres = [], []
    x = -total / 2
    for i, (enc, pts, width) in enumerate(per_model):
        slot_c = x + width / 2
        for w, ca, rel in pts:
            specs.append((w, enc, ca, slot_c + rel))
        slot_centres.append((enc, slot_c, width / 2))
        x += width + (SLOT_GAP if i < n_gaps else 0)

    return specs, slot_centres, total


_VIEW_SPECS = {v: _view_point_specs(v) for v in VIEW_ORDER}
_MAX_VIEW_WIDTH = max(total for _, _, total in _VIEW_SPECS.values())
VIEW_SEP = _MAX_VIEW_WIDTH + 0.45   # centre-to-centre distance between view groups

# Near-axis (model type) ticks, far-axis (view group) label/bracket positions, the
# light separators between model-type slots, and the slot background tiles are
# purely structural — identical for every panel — so build them once.
# dx+xf has only 3 single-point slots; give it a compact allocation and let
# the logsig groups fill the freed space (logsig centres shift left).
_dx_content  = _VIEW_SPECS[VIEW_ORDER[0]][2]   # = 2 * SLOT_GAP ≈ 0.50
_DX_HALF     = _dx_content / 2 + 0.30         # content half + small margin each side
_group_bounds = ([-_DX_HALF, _DX_HALF] +
                 [_DX_HALF + k * VIEW_SEP for k in range(1, len(VIEW_ORDER) + 1)])
view_centres  = [(_group_bounds[i] + _group_bounds[i + 1]) / 2
                 for i in range(len(VIEW_ORDER))]
_slot_ticks, _slot_labels = [], []
_slot_tiles  = []      # (lo, hi) per slot, abutting neighbours — for zebra shading
_slot_separators = []  # x-positions of the gap midpoint between adjacent slots
_group_ranges = []     # (view_key, xc, half_width)
for vi, view_key in enumerate(VIEW_ORDER):
    xc = view_centres[vi]
    _specs, _slot_centres, _total = _VIEW_SPECS[view_key]
    _group_lo, _group_hi = _group_bounds[vi], _group_bounds[vi + 1]
    _seps_in_group = [xc + ((c0 + h0) + (c1 - h1)) / 2
                       for (_, c0, h0), (_, c1, h1) in zip(_slot_centres, _slot_centres[1:])]
    _edges = [_group_lo] + _seps_in_group + [_group_hi]
    for i, (enc, slot_c, slot_half) in enumerate(_slot_centres):
        _slot_ticks.append(xc + slot_c)
        _slot_labels.append(ENC_AXIS_LABEL[enc])
        _slot_tiles.append((_edges[i], _edges[i + 1]))
    _slot_separators.extend(_seps_in_group)
    _group_ranges.append((view_key, xc, _total / 2))

# ---------------------------------------------------------------------------
# SimMTM helper
# ---------------------------------------------------------------------------

def read_simmtm_tsv(path: str, metric: str = 'accuracy'):
    """Return dict (pretrain_dataset, target_dataset) → list[float].

    TSV columns: pretrain_dataset target_dataset seed epochs_pretrain epochs_finetune accuracy f1_score
    metric: 'accuracy' (col 5) or 'f1_macro' (col 6, stored as f1_score in the TSV).
    """
    col = 6 if metric == 'f1_macro' else 5
    scores = {}
    try:
        with open(path) as f:
            header = None
            for line in f:
                line = line.rstrip('\n')
                if not line or line.startswith('#'):
                    continue
                parts = line.split('\t')
                if header is None:
                    header = parts
                    continue
                if len(parts) <= col:
                    continue
                try:
                    val = float(parts[col])
                except ValueError:
                    continue
                scores.setdefault((parts[0], parts[1]), []).append(val)
    except FileNotFoundError:
        pass
    return scores

# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def parse_run_row(name: str):
    """Return (view_key, window, encoder, channel_adapt) or None.

    Shared core for any run name (finetune/baseline/freeze) — suffix and
    pretrain-source filtering are the caller's responsibility.
    """
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
    # bare 2-view tag (v2logsig) → canonical key (v2logsig_nview)
    if '_v3' not in view_key and not view_key.endswith('_nview'):
        view_key = view_key + '_nview'
    if view_key not in VIEW_LABEL:
        return None

    has_lead_lag = re.search(r'_ll\d+', name) is not None
    if '_win64_norm' in name:
        # normalised win64, optionally combined with lead-lag (channel-adapt trials)
        # or the (separate, older) logsig_lag mechanism.
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
        # logsig_global_time / logsig_pool ablation (mechanistic depth/gt/pool sweep)
        if '_gt_' in name:
            window = window + '_gt'
        if '_plast' in name:
            window = window + '_plast'
        depth_m = re.search(r'_d(\d+)_', name)
        if depth_m:
            window = window + f'_d{depth_m.group(1)}'
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

    return view_key, window, encoder, channel_adapt


def parse_finetune_row(name: str):
    """Return (view_key, window, encoder, channel_adapt) or None.
    Only capture24_256_00 pretrain — this file is scoped to one pretrain source."""
    if not name or not name.endswith('_finetune'):
        return None
    if PRETRAIN_TAG not in name:
        return None
    # The dx+xf baseline was swept at ep2 (the comparable, many-seed set); the
    # sparse ep200 dx+xf reruns are one-off and not part of that comparison.
    m_ep = re.search(r'v2dx_v3xf_ep(\d+)_', name)
    if m_ep and m_ep.group(1) != '2':
        return None
    return parse_run_row(name)


def read_tsv(path: str, parser=parse_finetune_row):
    """Return dict (view_key, window, encoder, channel_adapt) → list[float]."""
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
                parsed = parser(parts[0])
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


def plot_dataset(ax, scores, simmtm_vals, title, metric_label, baseline_mean=None):
    """Draw one categorical-scatter panel for a single finetune target dataset."""

    # alternating zebra shading per model-type slot (greyscale)
    for si, (lo, hi) in enumerate(_slot_tiles):
        if si % 2 == 1:
            ax.axvspan(lo, hi, color='#f0f0f0', zorder=0, linewidth=0)

    for vi, view_key in enumerate(VIEW_ORDER):
        xc = view_centres[vi]
        specs, _, _ = _VIEW_SPECS[view_key]

        for win, enc, ca, rel_x in specs:
            x = xc + rel_x

            if enc == 'simmtm':
                if not simmtm_vals:
                    continue
                mu, ci = _mean_ci(simmtm_vals)
                yerr = None if ci is None else [[min(ci, mu)], [min(ci, 1.0 - mu)]]
                ax.errorbar(x, mu, yerr=yerr,
                            fmt=SIMMTM_MARKER, markersize=SIMMTM_SIZE,
                            markerfacecolor=SIMMTM_COLOR, markeredgecolor=SIMMTM_COLOR,
                            ecolor=SIMMTM_COLOR, elinewidth=1.1, capsize=2, zorder=5)
                continue

            vals = scores.get((view_key, win, enc, ca), [])
            mu, ci = _mean_ci(vals)
            if mu is None:
                continue
            method = MAJOR_METHOD[win]
            adj    = ADJUSTMENT[win]

            yerr = None if ci is None else [[min(ci, mu)], [min(ci, 1.0 - mu)]]
            ax.errorbar(x, mu, yerr=yerr,
                        fmt=METHOD_MARKER[method],
                        markersize=MARKER_SIZE,
                        markerfacecolor=ADJUSTMENT_COLOR[adj],
                        markeredgecolor='#333',
                        markeredgewidth=0.6,
                        ecolor='#333',
                        elinewidth=1.1,
                        capsize=2,
                        zorder=5)

            overlay = CHADAPT_OVERLAY[ca]
            if overlay is not None:
                ax.plot(x, mu, marker=overlay, color='black',
                        markersize=4.2, markeredgewidth=1.3, zorder=6)

        if vi < len(VIEW_ORDER) - 1:
            ax.axvline(_group_bounds[vi + 1], color='#999999',
                       linewidth=1.0, linestyle='-', zorder=1)

    # light separators between model-type slots
    for sep_x in _slot_separators:
        ax.axvline(sep_x, color='#cccccc', linewidth=0.6, linestyle=':', zorder=1)

    # Baseline reference line
    if baseline_mean is not None:
        ax.axhline(baseline_mean, color='black', linestyle=':', linewidth=1.1, zorder=4)
        ax.text(view_centres[-1] + VIEW_SEP * 0.45, baseline_mean + 0.004,
                f'ref {baseline_mean:.3f}', fontsize=6.5, va='bottom', ha='right', color='#222')

    # Near-axis row: model type, at every slot centre.
    ax.set_xticks(_slot_ticks)
    ax.set_xticklabels(_slot_labels, fontsize=6.3, rotation=50, ha='right')
    ax.tick_params(axis='x', pad=2)

    # Far-axis row: view-combo group label + bracket, in axes-fraction y so it
    # tracks the data x-range but sits at a fixed vertical offset.
    trans = blended_transform_factory(ax.transData, ax.transAxes)
    bracket_y, label_y = -0.42, -0.47
    for vk, gxc, ghalf in _group_ranges:
        ax.plot([gxc - ghalf, gxc + ghalf], [bracket_y, bracket_y],
                transform=trans, color='#555', linewidth=0.9,
                clip_on=False, zorder=2)
        ax.annotate(VIEW_LABEL[vk], xy=(gxc, label_y), xycoords=trans,
                    ha='center', va='top', fontsize=7, color='#222',
                    annotation_clip=False)

    ax.set_xlim(_group_bounds[0] - 0.1, _group_bounds[-1] + 0.1)
    ax.set_ylabel(metric_label, fontsize=9)
    ax.set_title(title, fontsize=10, fontweight='bold')
    ax.set_ylim(0.7, 1.05)
    ax.yaxis.grid(True, linewidth=0.4, alpha=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.spines[['top', 'right']].set_visible(False)

    # Seed count annotation on the baseline view group
    cx0 = view_centres[0]
    all_n = [len(scores.get((VIEW_ORDER[0], 'global', e, 'none'), []))
             for e in ['transformer', 'transformer_bilinear']]
    n = max(all_n) if all_n else 0
    if n > 0:
        ax.text(cx0, 0.72, f'n={n}', ha='center', va='bottom',
                fontsize=7, color='gray')


def build_legend(fig):
    """Shared legend: major method (shape), adjustment (colour), channel adapt (overlay)."""
    method_handles = [
        Line2D([0], [0], marker=METHOD_MARKER[m], linestyle='none',
               markerfacecolor='#888', markeredgecolor='#333', markersize=8,
               label=METHOD_LABEL[m])
        for m in METHOD_ORDER
    ] + [
        Line2D([0], [0], marker=SIMMTM_MARKER, linestyle='none',
               markerfacecolor=SIMMTM_COLOR, markeredgecolor=SIMMTM_COLOR,
               markersize=10, label='SimMTM'),
    ]
    adjustment_handles = [
        Line2D([0], [0], marker='o', linestyle='none',
               markerfacecolor=ADJUSTMENT_COLOR[a], markeredgecolor='#333', markersize=8,
               label=ADJUSTMENT_LABEL[a])
        for a in ADJUSTMENT_ORDER
    ]
    chadapt_handles = [
        Line2D([0], [0], marker='o', linestyle='none', markerfacecolor='#ccc',
               markeredgecolor='#333', markersize=7, label=CHADAPT_LABEL['none']),
        Line2D([0], [0], marker='.', linestyle='none', markerfacecolor='black',
               markeredgecolor='black', markersize=11, label=CHADAPT_LABEL['drop']),
        Line2D([0], [0], marker='x', linestyle='none', markerfacecolor='black',
               markeredgecolor='black', markersize=7, label=CHADAPT_LABEL['pca']),
        Line2D([0], [0], marker='+', linestyle='none', markerfacecolor='black',
               markeredgecolor='black', markersize=7, markeredgewidth=1.4, label=CHADAPT_LABEL['copy']),
    ]

    leg1 = fig.legend(handles=method_handles,
                      loc='lower center', bbox_to_anchor=(0.16, -0.04),
                      ncol=4, fontsize=8, frameon=False,
                      title='Major method / comparison', title_fontsize=8.5)
    leg2 = fig.legend(handles=adjustment_handles,
                      loc='lower center', bbox_to_anchor=(0.58, -0.04),
                      ncol=5, fontsize=8, frameon=False,
                      title='Adjustment', title_fontsize=8.5)
    fig.legend(handles=chadapt_handles,
               loc='lower center', bbox_to_anchor=(0.92, -0.04),
               ncol=1, fontsize=8, frameon=False,
               title='Channel adapt', title_fontsize=8.5)
    fig.add_artist(leg1)
    fig.add_artist(leg2)


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
                        choices=['accuracy', 'f1_macro'],
                        help='Metric column to visualise')
    parser.add_argument('--out', default='plots/capture24_results.pdf',
                        help='Output file (pdf/png/svg)')
    args = parser.parse_args()

    _simmtm_scores = read_simmtm_tsv(SIMMTM_TSV, metric=args.metric)

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
        simmtm_vals = _simmtm_scores.get(('capture24', short), [])
        panels.append((label, scores, simmtm_vals))

    if not panels:
        print('No data found for any requested dataset.')
        return

    metric_label = 'Test accuracy' if args.metric == 'accuracy' else 'Test F1'

    ncols = min(2, len(panels))
    nrows = (len(panels) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(13 * ncols, 6.5 * nrows),
                             squeeze=False)

    for idx, (label, scores, simmtm_vals) in enumerate(panels):
        ax = axes[idx // ncols][idx % ncols]
        # Prefer attention baseline; fall back to bilinear if not yet run
        baseline_vals = scores.get(('v2dx_v3xf', 'global', 'transformer', 'none'), [])
        if not baseline_vals:
            baseline_vals = scores.get(('v2dx_v3xf', 'global', 'transformer_bilinear', 'none'), [])
        baseline_mean, _ = _mean_ci(baseline_vals)
        plot_dataset(ax, scores, simmtm_vals, f'capture24 → {label}', metric_label, baseline_mean)

    # Hide unused subplots
    for idx in range(len(panels), nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    build_legend(fig)
    fig.suptitle(f'Multi-View Contrastive Learning — capture24 pretrain ({metric_label})',
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

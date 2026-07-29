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

import os
import re
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from matplotlib.transforms import blended_transform_factory
from scipy import stats

matplotlib.rcParams['hatch.linewidth'] = 2.0   # default 1.0 is invisible on thin bars

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Comment out any entry here to hide it from all plots.
DISABLED_WINDOWS = {
    # 'global',
    # 'win64',
    # 'win64_sl1',
    # 'win64_ll',
    # 'win64_rp',
    # 'win64_ll_rp',
    # 'norm_win64',
    # 'norm_win64_lag',
    # 'win128',
    # 'win128_s7',
    # 'tukey64',
    # 'tukey128',
    # 'tukey128_s7',
}

VIEW_ORDER = ['v2dx_v3xf', 'v2logsig_v3xf', 'v2dx_v3logsig', 'v2logsig_nview']
VIEW_LABEL = {
    'v2dx_v3xf':      'dx + xf\n(baseline)',
    'v2logsig_v3xf':  'logsig + xf',
    'v2dx_v3logsig':  'dx + logsig',
    'v2logsig_nview': 'logsig\n(2-view)',
}

_ALL_WINDOW_ORDER = ['global', 'win64', 'win64_sl1', 'win64_ll', 'win64_rp', 'win64_ll_rp',
                     'win64_norm_ll', 'norm_win64', 'norm_win64_lag', 'win128', 'win128_s7',
                     'tukey64', 'tukey128', 'tukey128_s7']
WINDOW_ORDER = [w for w in _ALL_WINDOW_ORDER if w not in DISABLED_WINDOWS]
WINDOW_LABEL = {
    'global':        'global',
    'win64':         'win 64',
    'win64_sl1':     'win 64\n(no lv1)',
    'win64_ll':      'win 64\n+lead-lag',
    'win64_rp':      'win 64\n+red. pen.',
    'win64_ll_rp':   'win 64\n+ll+rp',
    'win64_norm_ll': 'win 64\n+norm+ll',
    'norm_win64':    'norm win 64',
    'norm_win64_lag':'norm win 64\n+lag',
    'win128':        'win 128',
    'win128_s7':     'win 128\ns7',
    'tukey64':       'tukey 64',
    'tukey128':      'tukey 128',
    'tukey128_s7':   'tukey 128\ns7',
}
_cmap = matplotlib.colormaps['nipy_spectral']
WINDOW_COLOR = {w: _cmap((i + 0.5) / len(_ALL_WINDOW_ORDER))
                for i, w in enumerate(_ALL_WINDOW_ORDER)}

# ---------------------------------------------------------------------------
# Figure 1 vocabulary: major method (marker shape) x adjustment (marker colour)
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
    'norm_win64':     'norm_win64',
    'norm_win64_lag': 'norm_win64',
    'win128':         'win128',
    'win128_s7':      'win128',
    'tukey64':        'tukey64',
    'tukey128':       'tukey128',
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
    'norm_win64':     'none',
    'norm_win64_lag': 'lag',
    'win128':         'none',
    'win128_s7':      'stride7',
    'tukey64':        'none',
    'tukey128':       'none',
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

# Order (and therefore color index) must match visualize_capture24.py's
# ADJUSTMENT_ORDER exactly, so the same adjustment always gets the same color
# across both files — 'noise' has no data here (win64_lsn0.1 isn't part of
# this file's window set) but is kept as a placeholder to preserve alignment.
ADJUSTMENT_ORDER = ['none', 'no_lv1', 'lead_lag', 'reduced_pen', 'lead_lag_rp',
                    'norm_leadlag', 'noise', 'lag', 'stride7']
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
}
_tab10 = matplotlib.colormaps['tab10'].colors
ADJUSTMENT_COLOR = {adj: _tab10[i % len(_tab10)] for i, adj in enumerate(ADJUSTMENT_ORDER)}

# channel_adapt: overlay drawn on top of the base (method, adjustment) marker, so the
# channel-adapt family still reads as "the same win64 point, with a mark on it".
CHADAPT_ORDER = ['none', 'drop', 'pca', 'copy']
CHADAPT_OVERLAY = {'none': None, 'drop': '.', 'pca': 'x', 'copy': '+'}
CHADAPT_LABEL = {'none': 'none (no mark)', 'drop': 'drop (• overlay)',
                 'pca': 'pca (× overlay)', 'copy': 'copy (+ overlay)'}

# Three encoder variants:
#   transformer       — default, mean pool over logsig time dim (auto)
#   transformer_plast — ablation: last-token pool (--logsig_pool last)
#   mlp_logsig        — MLP branch, last-token pool (auto for stream)
ENCODERS  = ['transformer', 'transformer_plast', 'mlp_logsig',
             'mlp_logsig_bilinear', 'transformer_bilinear']
ENC_LABEL = {
    'transformer':          'Transformer (mean pool)',
    'transformer_plast':    'Transformer (last pool)',
    'mlp_logsig':           'MLP-LogSig (last pool)',
    'mlp_logsig_bilinear':  'MLP-LogSig bilinear',
    'transformer_bilinear': 'Transformer bilinear',
}
# Short form used for the close-to-axis "model type" tick row in Figure 1.
ENC_AXIS_LABEL = {
    'transformer':          'mean pool',
    'transformer_plast':    'last pool',
    'mlp_logsig':           'MLP-LogSig',
    'mlp_logsig_bilinear':  'MLP-LogSig+bil',
    'transformer_bilinear': 'Transformer+bil',
    'simmtm':               'SimMTM',
}

# ---------------------------------------------------------------------------
# Point geometry
#
# Within a view group, points are organised model-type-major: each encoder/
# pooling variant ("model type") gets its own x-slot (in ENCODERS order), and
# within a slot every (window) point gets its own x-position, clustered by
# major method. The one exception is win64_norm_ll under mlp_logsig_bilinear,
# which expands into 3 tightly-spaced channel_adapt sub-positions (the
# channel-adapt trial suite varies channel_adapt at fixed window config).
# ---------------------------------------------------------------------------

POINT_PITCH          = 0.080   # centre-to-centre spacing between adjacent points (same method)
BIL_POINT_PITCH      = 0.108   # wider spacing within mlp_logsig_bilinear
COMPACT_POINT_PITCH  = 0.018   # tight spacing for transformer (mean-pool) slot — halves its tile
COMPACT_METHOD_GAP   = 0.050   # compressed inter-method gap within the transformer slot
METHOD_GAP      = 0.100   # extra gap injected when the major method changes
SLOT_GAP        = 0.25    # gap between model-type slots within a view group
CHADAPT_PITCH   = 0.044   # tight centre-to-centre spacing for channel_adapt sub-positions
MARKER_SIZE     = 6.0

# win64, win128, tukey128 (+ the channel-adapt win64_norm_ll combo) have bilinear data.
_BIL_WIN_SET = {'win64', 'win64_sl1', 'win64_ll', 'win64_rp', 'win64_ll_rp',
                'win64_norm_ll', 'norm_win64', 'win128', 'tukey128'}


def _model_types_for_window(win):
    if win == 'global':
        return ['transformer', 'transformer_plast', 'mlp_logsig']
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
        if w == 'win64_norm_ll' and model_type == 'mlp_logsig_bilinear':
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
    slot_centres = [(model_type, rel_x_centre, half_width), ...] for the near-axis tick
                   row and the slot-boundary separators (half_width is the point-cluster
                   half-width, not including the surrounding SLOT_GAP padding)
    """
    if view_key == 'v2dx_v3xf':
        per_model = [('transformer', [('global', 'none', 0.0)], 0.0),
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

# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def parse_finetune_row(name: str, expected_pretrain: str = None):
    """
    Returns (view_key, window, encoder, channel_adapt) for finetune rows, else None.
    Skips probe rows, non-finetune suffixes, cross-time and view-embed variants, and
    (when expected_pretrain is given) rows pretrained on a different dataset — the
    same TSV can hold runs pretrained on different source datasets (e.g. HARTH and
    capture24 both finetune onto HAR70plus), and they must not be conflated.
    Bilinear variants are included (mlp_logsig_bilinear / transformer_bilinear).
    """
    if not name or not name.endswith('_finetune') or name.startswith('probe_'):
        return None
    # Cross-time and view-embed interaction variants belong to a separate plot.
    if any(m in name for m in ('_ilcrosstime_', '_ilviewembed_')):
        return None

    if expected_pretrain is not None:
        pt = re.search(r'_pt-(.+?)_v2', name)
        if not pt or pt.group(1) != expected_pretrain:
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


# ---------------------------------------------------------------------------
# SimMTM comparison baseline
# ---------------------------------------------------------------------------

SIMMTM_TSV = 'SimMTM/SimMTM_Classification/results/simmtm_results.tsv'
SIMMTM_MARKER = '*'
SIMMTM_COLOR  = '#222222'
SIMMTM_SIZE   = MARKER_SIZE * 0.8


def read_simmtm_tsv(path: str):
    """Return dict (pretrain_dataset, target_dataset) → list[float] (accuracy)."""
    scores = {}
    try:
        with open(path) as f:
            for line in f:
                line = line.rstrip('\n')
                if not line or line.startswith('#') or line.startswith('pretrain'):
                    continue
                parts = line.split('\t')
                if len(parts) < 6:
                    continue
                try:
                    acc = float(parts[5])
                except ValueError:
                    continue
                scores.setdefault((parts[0], parts[1]), []).append(acc)
    except FileNotFoundError:
        pass
    return scores


_simmtm_scores = read_simmtm_tsv(SIMMTM_TSV)

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


def read_tsv(*paths: str, expected_pretrain: str = None):
    """Return dict (view_key, window, encoder, channel_adapt) → list[float] for
    finetune rows. Accepts multiple paths; results are merged (lists concatenated).
    expected_pretrain, if given, drops rows pretrained on any other dataset."""
    scores = {}
    for path in paths:
        try:
            with open(path) as f:
                for line in f:
                    line = line.rstrip('\n')
                    if not line or line.startswith('#'):
                        continue
                    parts = line.split('\t')
                    if len(parts) < 2:
                        continue
                    parsed = parse_finetune_row(parts[0], expected_pretrain=expected_pretrain)
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


def read_bilinear_tsv(*paths: str):
    """Return dict (view_key, window, il) → list[float].
    Accepts multiple paths; results are merged (lists concatenated)."""
    scores = {}
    for path in paths:
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
    'out_finetune/_DA_Epilepsy_256_00/final_test_metric_summary.tsv',
    'out_finetune_pre_canada_old/_DA_Epilepsy_256_00/final_test_metric_summary.tsv',
    expected_pretrain='_DA_SleepEEG_256_00')
har_scores = read_tsv(
    'out_finetune/_DA_HAR70plus_256_00/final_test_metric_summary.tsv',
    'out_finetune_pre_canada_old/_DA_HAR70plus_256_00/final_test_metric_summary.tsv',
    expected_pretrain='_DA_HARTH_256_00')

panels = [
    ('Epilepsy  (SleepEEG pretrain)', epilepsy_scores,
     _simmtm_scores.get(('SleepEEG', 'Epilepsy'), [])),
    ('HAR70plus  (HARTH pretrain)',   har_scores,
     _simmtm_scores.get(('HARTH', 'HAR70plus'), [])),
]

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

fig, axes = plt.subplots(1, 2, figsize=(24, 7.5), sharey=False)
fig.suptitle('Finetune accuracy — view combinations, logsig variants, encoders',
             fontsize=13, fontweight='bold', y=1.01)

# dx+xf has only 3 single-point slots; give it a compact allocation and let
# the logsig groups fill the freed space (logsig centres shift left).
_dx_content  = _VIEW_SPECS[VIEW_ORDER[0]][2]   # = 2 * SLOT_GAP ≈ 0.50
_DX_HALF     = _dx_content / 2 + 0.30         # content half + small margin each side
_group_bounds = ([-_DX_HALF, _DX_HALF] +
                 [_DX_HALF + k * VIEW_SEP for k in range(1, len(VIEW_ORDER) + 1)])
view_centres  = [(_group_bounds[i] + _group_bounds[i + 1]) / 2
                 for i in range(len(VIEW_ORDER))]

# Near-axis (model type) ticks, far-axis (view group) label/bracket positions, the
# light separators between model-type slots, and the slot background tiles are
# purely structural — identical for both panels — so build them once.
_slot_ticks, _slot_labels = [], []
_slot_tiles  = []      # (lo, hi) per slot, abutting neighbours — for zebra shading
_slot_separators = []  # x-positions of the gap midpoint between adjacent slots
_group_ranges = []  # (view_key, xc, half_width)
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

for ax, (title, scores, simmtm_vals) in zip(axes, panels):

    ref_vals  = scores.get(('v2dx_v3xf', 'global', 'transformer', 'none'), [])
    ref_score = np.mean(ref_vals) if ref_vals else None

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
                score = np.mean(simmtm_vals)
                yerr = None
                if len(simmtm_vals) > 1:
                    n    = len(simmtm_vals)
                    std  = np.std(simmtm_vals, ddof=1)
                    ci95 = stats.t.ppf(0.975, df=n - 1) * std / np.sqrt(n)
                    yerr = [[min(ci95, score)], [min(ci95, 1.0 - score)]]
                ax.errorbar(x, score, yerr=yerr,
                            fmt=SIMMTM_MARKER, markersize=SIMMTM_SIZE,
                            markerfacecolor=SIMMTM_COLOR, markeredgecolor=SIMMTM_COLOR,
                            ecolor=SIMMTM_COLOR, elinewidth=1.3, capsize=2.5, zorder=5)
                continue

            key  = (view_key, win, enc, ca)
            vals = scores.get(key, [])
            if not vals:
                continue
            score  = np.mean(vals)
            method = MAJOR_METHOD[win]
            adj    = ADJUSTMENT[win]

            yerr = None
            if len(vals) > 1:
                n    = len(vals)
                std  = np.std(vals, ddof=1)
                ci95 = stats.t.ppf(0.975, df=n - 1) * std / np.sqrt(n)
                yerr = [[min(ci95, score)], [min(ci95, 1.0 - score)]]

            ax.errorbar(x, score, yerr=yerr,
                        fmt=METHOD_MARKER[method],
                        markersize=MARKER_SIZE,
                        markerfacecolor=ADJUSTMENT_COLOR[adj],
                        markeredgecolor='#333',
                        markeredgewidth=0.6,
                        ecolor='#333',
                        elinewidth=1.3,
                        capsize=2.5,
                        zorder=5)

            overlay = CHADAPT_OVERLAY[ca]
            if overlay is not None:
                ax.plot(x, score, marker=overlay, color='black',
                        markersize=4.5, markeredgewidth=1.4, zorder=6)

        if vi < len(VIEW_ORDER) - 1:
            ax.axvline(_group_bounds[vi + 1], color='#999999',
                       linewidth=1.2, linestyle='-', zorder=1)

    # light separators between model-type slots, so each architecture's point
    # cluster reads as its own column rather than bleeding into its neighbour
    for sep_x in _slot_separators:
        ax.axvline(sep_x, color='#cccccc', linewidth=0.7, linestyle=':', zorder=1)

    if ref_score is not None:
        ax.axhline(ref_score, color='#222', linewidth=1.2,
                   linestyle=':', zorder=4)
        ax.text(view_centres[-1] + VIEW_SEP * 0.45,
                ref_score + 0.002,
                f'ref {ref_score:.3f}',
                fontsize=7.5, va='bottom', ha='right', color='#222')

    ax.set_title(title, fontsize=10, pad=8)

    # Near-axis row: model type, at every slot centre.
    ax.set_xticks(_slot_ticks)
    ax.set_xticklabels(_slot_labels, fontsize=7.5, rotation=50, ha='right')
    ax.tick_params(axis='x', pad=2)

    trans = blended_transform_factory(ax.transData, ax.transAxes)

    # Far-axis row: view-combo group label + bracket, in axes-fraction y so it
    # tracks the data x-range but sits at a fixed vertical offset.
    bracket_y, label_y = -0.30, -0.335
    for view_key, gxc, ghalf in _group_ranges:
        ax.plot([gxc - ghalf, gxc + ghalf], [bracket_y, bracket_y],
                transform=trans, color='#555', linewidth=1.0,
                clip_on=False, zorder=2)
        ax.annotate(VIEW_LABEL[view_key], xy=(gxc, label_y), xycoords=trans,
                    ha='center', va='top', fontsize=8.5, color='#222',
                    annotation_clip=False)

    ax.set_ylabel('Accuracy', fontsize=9)

    all_scores = [np.mean(v) for v in scores.values() if v]
    ymin = max(0.0, min(all_scores) - 0.04) if all_scores else 0.0
    ax.set_ylim(ymin, 1.03)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.2f}'))
    ax.grid(axis='y', linestyle='--', alpha=0.4, zorder=0)
    ax.spines[['top', 'right']].set_visible(False)
    ax.set_xlim(_group_bounds[0] - 0.1, _group_bounds[-1] + 0.1)

# ---------------------------------------------------------------------------
# Legend
# ---------------------------------------------------------------------------

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
                  loc='lower center', bbox_to_anchor=(0.18, -0.14),
                  ncol=4, fontsize=8.5, frameon=False,
                  title='Major method / comparison', title_fontsize=9)
leg2 = fig.legend(handles=adjustment_handles,
                  loc='lower center', bbox_to_anchor=(0.58, -0.14),
                  ncol=4, fontsize=8.5, frameon=False,
                  title='Adjustment', title_fontsize=9)
leg3 = fig.legend(handles=chadapt_handles,
                  loc='lower center', bbox_to_anchor=(0.92, -0.14),
                  ncol=1, fontsize=8.5, frameon=False,
                  title='Channel adapt', title_fontsize=9)
fig.add_artist(leg1)
fig.add_artist(leg2)

plt.tight_layout()
os.makedirs('plots', exist_ok=True)
plt.savefig('plots/finetune_results.pdf', bbox_inches='tight')
plt.savefig('plots/finetune_results.png', dpi=150, bbox_inches='tight')
print('Saved: plots/finetune_results.pdf  plots/finetune_results.png')
#plt.show()

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
                _sem = np.std(vals, ddof=1) / len(vals) ** 0.5
                ax.errorbar(bar_x, score,
                            yerr=[[min(_sem, score)], [min(_sem, 1.0 - score)]],
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
plt.savefig('plots/interaction_type_results.pdf', bbox_inches='tight')
plt.savefig('plots/interaction_type_results.png', dpi=150, bbox_inches='tight')
print('Saved: plots/interaction_type_results.pdf  plots/interaction_type_results.png')
#plt.show()

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
    'out_finetune/_DA_Epilepsy_256_00/final_test_metric_summary.tsv',
    'out_finetune_pre_canada_old/_DA_Epilepsy_256_00/final_test_metric_summary.tsv')
bil_har = read_bilinear_tsv(
    'out_finetune/_DA_HAR70plus_256_00/final_test_metric_summary.tsv',
    'out_finetune_pre_canada_old/_DA_HAR70plus_256_00/final_test_metric_summary.tsv')

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
                                    yerr=[[min(ci95, score)], [min(ci95, 1.0 - score)]],
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
plt.savefig('plots/bilinear_logsig_results.pdf', bbox_inches='tight')
plt.savefig('plots/bilinear_logsig_results.png', dpi=150, bbox_inches='tight')
print('Saved: plots/bilinear_logsig_results.pdf  plots/bilinear_logsig_results.png')
#plt.show()

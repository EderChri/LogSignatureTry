"""nonstationarity_vs_signature_benefit.py

Tests whether the log-signature view's transfer benefit scales with a target
dataset's within-observation non-stationarity.

Result directories
-------------------
Results are scattered across out_finetune/, out_finetune_old/ and
out_finetune_pre_canada_old/ (pre-migration archives) — in particular the
HARTH->HAR70plus dx+xf baseline only survives in the archived directories, not
in the current out_finetune/. All final_test_metric_summary.tsv files under
all three roots are pooled, de-duplicated on the exact (run_name, score) pair
(archived copies repeat identical rows verbatim; genuinely repeated runs of
the same nominal tag with a different score are kept as separate samples).

Channel comparability
----------------------
Datasets differ wildly in channel count (WISDM: 3, Opportunity: 113) and a plain
fraction-of-channels-failing-a-test isn't comparable across them. Every HAR
dataset here has a genuine tri-axial accelerometer as its FIRST 3 channels:
  - HAR70plus:   back_x, back_y, back_z            (data_preprocess.py ACCEL_COLS)
  - WISDM/WISDM2: x, y, z                            (phone accelerometer)
  - USC_HAD:     acc_x, acc_y, acc_z                 (feature_column in USC_HAD_Loader.py)
  - Opportunity: first on-body IMU's acc x,y,z       (standard 113-ch DeepConvLSTM layout)
  - Skoda:       first sensor's calibrated acc x,y,z (6 cols/sensor in Skoda_Loader.py)
All non-stationarity magnitude below is computed only on channels [0, 1, 2],
averaged across the three raw axes (no magnitude-vector projection).

Non-stationarity magnitude (continuous, not a test-rejection fraction)
------------------------------------------------------------------------
Each window is split into 4 equal sub-blocks. We compute:
  - mean_drift: std of block means, normalized by the window's overall std
  - var_cv:     coefficient of variation of block variances
magnitude = mean(mean_drift, var_cv), averaged over sampled windows and the 3
accelerometer channels. Higher -> more within-observation drift. Computed once
per target dataset (independent of source/pretraining).

Transfer benefit (y-axis)
--------------------------
For each source -> target pair, delta = mean finetune test score of the
signature-view config minus the dx+xf baseline config. Both are held to a
single fixed architecture so every pair is an apples-to-apples comparison:
  base:      v2dx_v3xf,                    bilinear interaction
  signature: v2logsig_v3xf, mlp_logsig encoder, window logsig (size 64), bilinear interaction
This is the only signature configuration with full seed coverage on every one
of the 7 pairs (including HARTH->HAR70plus, which never got the plain-
transformer/stream-mode logsig variant run) — chosen over the previously used
plain-transformer/stream-mode logsig config specifically so HARTH could be
included instead of dropped.

Usage
-----
  python vis/nonstationarity_vs_signature_benefit.py
"""

import glob
import os
import pickle
import re
import zlib

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ACCEL_CHANNELS = [0, 1, 2]
N_BLOCKS = 4

RESULT_ROOTS = ['out_finetune', 'out_finetune_old', 'out_finetune_pre_canada_old']

PAIRS = [
    ('HARTH', 'HAR70plus'),
    ('capture24', 'HAR70plus'),
    ('capture24', 'WISDM'),
    ('capture24', 'WISDM2'),
    ('capture24', 'USC_HAD'),
    ('capture24', 'Opportunity'),
    ('capture24', 'Skoda'),
]

RUN_NAME_RE = re.compile(
    r'^_DA_(?P<target>.+?)_256_00(?:_old)?_pt-_DA_(?P<source>.+?)_256_00_(?P<rest>.+)$')
BASE_CFG_RE = re.compile(
    r'^v2dx_v3xf_ep\d+_(?P<seed>\d+)_ilbilinear_(hidden|latent)_[A-Z]+_[\d.]+_\d+_finetune$')

# Candidate signature-view configs, keyed by a short label used on the CLI.
# 'v3xf_win64': 3-view (dx replaced by logsig), mlp_logsig encoder, window-64 logsig.
#               Only variant with full seed coverage on all 7 pairs.
# 'nview_tukey128': 2-view (xt+logsig only, no xf), mlp_logsig encoder, tukey-128
#               smoothed logsig. Only has seeds 4-9 for Opportunity/USC_HAD.
SIG_CFG_PATTERNS = {
    'v3xf_win64': r'^v2logsig_v3xf_ep\d+_(?P<seed>\d+)_mlp_logsig_win64_ilbilinear_(hidden|latent)_[A-Z]+_[\d.]+_\d+_finetune$',
    'nview_tukey128': r'^v2logsig_nview_ep\d+_(?P<seed>\d+)_mlp_logsig_tukey128_ilbilinear_(hidden|latent)_[A-Z]+_[\d.]+_\d+_finetune$',
}

# ---------------------------------------------------------------------------
# Non-stationarity magnitude
# ---------------------------------------------------------------------------

def window_magnitude(v: np.ndarray, eps: float = 1e-8) -> float:
    """Continuous within-window non-stationarity score for one channel's window."""
    blocks = v.reshape(N_BLOCKS, -1)
    block_means = blocks.mean(axis=1)
    block_vars  = blocks.var(axis=1)
    mean_drift = block_means.std() / (v.std() + eps)
    var_cv     = block_vars.std() / (block_vars.mean() + eps)
    return float(0.5 * (mean_drift + var_cv))


def dataset_magnitude(name: str, max_windows: int, base_seed: int) -> float:
    """Deterministic regardless of call order: seeded from the dataset name,
    not a shared/advancing RNG."""
    with open(f'preprocessed_data/_DA_{name}_256_00.pkl', 'rb') as f:
        X = pickle.load(f)[0]   # [N, C, L]
    rng = np.random.default_rng(base_seed + zlib.crc32(name.encode()))
    n = min(max_windows, len(X))
    idx = rng.choice(len(X), n, replace=False)
    scores = [window_magnitude(X[i, ch, :])
              for i in idx for ch in ACCEL_CHANNELS]
    return float(np.mean(scores))

# ---------------------------------------------------------------------------
# Transfer benefit — pooled across all result directories
# ---------------------------------------------------------------------------

def load_all_rows():
    """(target, source, tag_rest, score) for every row in every summary file,
    de-duplicated on the exact (run_name, score) pair across all roots."""
    files = []
    for root in RESULT_ROOTS:
        files += glob.glob(f'{root}/**/final_test_metric_summary.tsv', recursive=True)

    seen = set()
    rows = []
    for fp in files:
        with open(fp) as fh:
            next(fh)
            for line in fh:
                parts = line.rstrip('\n').split('\t')
                if len(parts) != 3:
                    continue
                run_name, score, _epochs = parts
                key = (run_name, score)
                if key in seen:
                    continue
                seen.add(key)
                m = RUN_NAME_RE.match(run_name)
                if not m:
                    continue
                rows.append((m.group('target'), m.group('source'), m.group('rest'), float(score)))
    return rows


def pair_delta(rows, source: str, target: str, sig_cfg_re):
    """Mean base/signature scores over the seeds present in BOTH configs
    (paired comparison — avoids e.g. comparing a 10-seed base to a 6-seed
    signature config as if they were the same sample size)."""
    base_by_seed = {m.group('seed'): s for t, src, rest, s in rows
                    if t == target and src == source
                    for m in [BASE_CFG_RE.match(rest)] if m}
    sig_by_seed  = {m.group('seed'): s for t, src, rest, s in rows
                    if t == target and src == source
                    for m in [sig_cfg_re.match(rest)] if m}
    common = sorted(set(base_by_seed) & set(sig_by_seed))
    if not common:
        return None
    base_scores = [base_by_seed[k] for k in common]
    sig_scores  = [sig_by_seed[k] for k in common]
    base_mean = float(np.mean(base_scores))
    sig_mean  = float(np.mean(sig_scores))
    return {
        'base_mean': base_mean, 'sig_mean': sig_mean,
        'delta': sig_mean - base_mean,
        'n': len(common), 'seeds': common,
    }

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(sig_config: str = 'v3xf_win64', targets: list = None,
        max_windows: int = 300, seed: int = 0):
    rows = load_all_rows()
    sig_cfg_re = re.compile(SIG_CFG_PATTERNS[sig_config])

    pairs = PAIRS if not targets else [p for p in PAIRS if p[1] in targets]

    magnitude_cache = {}
    results = []
    for source, target in pairs:
        delta_info = pair_delta(rows, source, target, sig_cfg_re)
        if delta_info is None:
            print(f'[{source}->{target}] No matched base/signature runs found — skipping.')
            continue
        if target not in magnitude_cache:
            magnitude_cache[target] = dataset_magnitude(target, max_windows, seed)
        mag = magnitude_cache[target]
        results.append((source, target, mag, delta_info))
        print(f'[{source}->{target}] magnitude={mag:.4f}  '
              f'base={delta_info["base_mean"]:.4f}  sig={delta_info["sig_mean"]:.4f}  '
              f'delta={delta_info["delta"]:+.4f}  (n={delta_info["n"]} matched seeds '
              f'{delta_info["seeds"]})')

    if len(results) < 3:
        print(f'\nOnly {len(results)} pair(s) — not enough to plot a trend line. '
              f'sig_config={sig_config!r}')
        return

    labels = [f'{src}→{tgt}' for src, tgt, _, _ in results]
    x = np.array([r[2] for r in results])
    y = np.array([r[3]['delta'] for r in results])

    slope, intercept = np.polyfit(x, y, 1)
    r = float(np.corrcoef(x, y)[0, 1])

    info_lines = [
        f'{len(results)} source→target pairs (HAR70plus has 2: from HARTH and from capture24)',
        'x: mean(mean-drift, block-variance CV) on the 3-axis accelerometer,',
        '   averaged over channels + sampled windows (higher = more non-stationary)',
        f'y: mean finetune score, signature ({sig_config}) minus baseline (v2dx_v3xf),',
        '   both bilinear interaction, paired over matching seeds',
        f'Pearson r = {r:.3f}   slope = {slope:+.4f} per unit magnitude',
        'Positive slope -> signature benefit grows with target non-stationarity.',
    ]

    os.makedirs('plots', exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.5, 5.8))

    point_color = '#4c72b0'
    line_color  = '#55555580'

    xs_line = np.linspace(x.min() - 0.02, x.max() + 0.02, 50)
    ax.plot(xs_line, slope * xs_line + intercept, color=line_color, lw=1.5,
            ls='--', zorder=1)
    ax.axhline(0, color='#888888', lw=1.0, ls=':', zorder=1)

    ax.scatter(x, y, s=70, color=point_color, edgecolor='white', linewidth=0.8,
               zorder=2)
    for label, xi, yi in zip(labels, x, y):
        ax.annotate(label, (xi, yi), textcoords='offset points', xytext=(6, 6),
                    fontsize=8.5)

    ax.set_xlabel('Target non-stationarity magnitude (accelerometer triad)', fontsize=10)
    ax.set_ylabel('Signature-model − base-model finetune accuracy', fontsize=10)
    ax.set_title('Signature benefit vs. target non-stationarity (all HAR pairs)',
                 fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.2)

    box_text = '\n'.join(info_lines)
    ax.text(0.98, 0.98, box_text, transform=ax.transAxes, fontsize=7.5,
            va='top', ha='right',
            bbox=dict(boxstyle='round', facecolor='#f5f5f5', edgecolor='#cccccc',
                       alpha=0.9))

    fig.tight_layout()
    out_png = 'plots/nonstationarity_vs_signature_benefit.png'
    fig.savefig(out_png, dpi=150)
    plt.close(fig)

    print('\n' + '\n'.join(info_lines))
    print(f'\nSaved plot -> {out_png}')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--sig-config', choices=list(SIG_CFG_PATTERNS), default='v3xf_win64')
    parser.add_argument('--targets', nargs='+', default=None,
                        help='Restrict to pairs with these target dataset names '
                             '(default: all 7 pairs)')
    args = parser.parse_args()
    run(sig_config=args.sig_config, targets=args.targets)

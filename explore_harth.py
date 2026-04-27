"""explore_harth.py — visual and statistical exploration of HARTH / HAR70plus.

Produces a PDF per dataset with:
  1. Class distribution (window counts).
  2. Sample windows per activity — one row per label.
  3. Rolling mean / rolling std on one subject's full recording.
  4. Per-window mean and std distributions, split by label.
  5. ADF + KPSS stationarity tests: fraction of windows where test rejects /
     fails to reject per channel and label.

Usage
-----
  python explore_harth.py                         # both datasets
  python explore_harth.py --dataset harth
  python explore_harth.py --dataset har70plus
  python explore_harth.py --dataset harth --subject S006.csv

The script reads raw CSV files from harth/ and har70plus/.
No preprocessed pickle files are required.

Dependencies: numpy, pandas, matplotlib, scipy, statsmodels
"""

import argparse
import os
import glob
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_pdf import PdfPages

from statsmodels.tsa.stattools import adfuller, kpss

# ---------------------------------------------------------------------------
# Dataset configuration
# ---------------------------------------------------------------------------

ACCEL_COLS = ['back_x', 'back_y', 'back_z', 'thigh_x', 'thigh_y', 'thigh_z']
WINDOW_LEN = 256
STRIDE     = 128
SAMPLE_HZ  = 50   # HARTH and HAR70plus are both 50 Hz

HARTH_LABEL_NAMES = {
    1: 'walking', 2: 'running', 3: 'shuffling',
    4: 'stairs↑', 5: 'stairs↓', 6: 'standing',
    7: 'sitting', 8: 'lying', 13: 'cycling(sit)',
    14: 'cycling(stand)', 130: 'cycling(sit,inact)', 140: 'cycling(stand,inact)',
}
HAR70_LABEL_NAMES = {
    1: 'walking', 3: 'shuffling', 4: 'stairs↑',
    5: 'stairs↓', 6: 'standing', 7: 'sitting', 8: 'lying',
}

DATASETS = {
    'harth':     ('harth',     HARTH_LABEL_NAMES,  set(HARTH_LABEL_NAMES)),
    'har70plus': ('har70plus', HAR70_LABEL_NAMES,  set(HAR70_LABEL_NAMES)),
}

# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def load_subject(fpath: str, keep_labels: set) -> pd.DataFrame:
    df = pd.read_csv(fpath, usecols=ACCEL_COLS + ['label'])
    df = df[df['label'].isin(keep_labels)].reset_index(drop=True)
    return df


def window_subject(df: pd.DataFrame, keep_labels: set):
    """Return (windows [N, L, 6], labels [N]) from a single subject's DataFrame."""
    values = df[ACCEL_COLS].values.astype(np.float64)
    labels = df['label'].values
    xs, ys = [], []
    for start in range(0, len(values) - WINDOW_LEN + 1, STRIDE):
        end   = start + WINDOW_LEN
        uq    = set(labels[start:end])
        if len(uq) == 1 and next(iter(uq)) in keep_labels:
            xs.append(values[start:end])   # [L, 6]
            ys.append(int(labels[start]))
    if not xs:
        return np.empty((0, WINDOW_LEN, 6)), np.empty(0, dtype=int)
    return np.stack(xs), np.array(ys, dtype=int)


def load_all_windows(data_dir: str, keep_labels: set):
    """Pool windows from all subjects in data_dir."""
    files = sorted(glob.glob(os.path.join(data_dir, '*.csv')))
    all_x, all_y = [], []
    for fp in files:
        df = load_subject(fp, keep_labels)
        if len(df) < WINDOW_LEN:
            continue
        x, y = window_subject(df, keep_labels)
        if len(x):
            all_x.append(x)
            all_y.append(y)
    return np.concatenate(all_x), np.concatenate(all_y)

# ---------------------------------------------------------------------------
# Stationarity tests
# ---------------------------------------------------------------------------

def adf_pvalue(series: np.ndarray) -> float:
    """ADF test p-value. Low p → reject unit root → stationary."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            result = adfuller(series, autolag='AIC', maxlag=10)
        return float(result[1])
    except Exception:
        return float('nan')


def kpss_pvalue(series: np.ndarray) -> float:
    """KPSS test p-value. Low p → reject stationarity → non-stationary."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            result = kpss(series, regression='c', nlags='auto')
        return float(result[1])
    except Exception:
        return float('nan')


def stationarity_summary(X: np.ndarray, y: np.ndarray,
                         label_names: dict, alpha: float = 0.05):
    """Compute per-channel, per-label stationarity statistics.

    Returns two dicts keyed by label with arrays of shape [num_channels]:
        adf_reject_frac  — fraction of windows where ADF rejects unit root (p<alpha)
                           → fraction that appear stationary by ADF
        kpss_reject_frac — fraction where KPSS rejects stationarity (p<alpha)
                           → fraction that appear non-stationary by KPSS
    """
    labels = sorted(label_names)
    adf_frac  = {}
    kpss_frac = {}

    for lbl in labels:
        mask   = y == lbl
        X_lbl  = X[mask]             # [N_lbl, L, 6]
        if len(X_lbl) == 0:
            continue
        n_ch   = X_lbl.shape[2]
        adf_r  = np.zeros(n_ch)
        kpss_r = np.zeros(n_ch)
        for ch in range(n_ch):
            adf_ps  = [adf_pvalue(X_lbl[i, :, ch])  for i in range(len(X_lbl))]
            kpss_ps = [kpss_pvalue(X_lbl[i, :, ch]) for i in range(len(X_lbl))]
            adf_r[ch]  = np.nanmean(np.array(adf_ps)  < alpha)
            kpss_r[ch] = np.nanmean(np.array(kpss_ps) < alpha)
        adf_frac[lbl]  = adf_r
        kpss_frac[lbl] = kpss_r

    return adf_frac, kpss_frac

# ---------------------------------------------------------------------------
# Plotting functions
# ---------------------------------------------------------------------------

def _tight(fig):
    fig.tight_layout()


def plot_class_distribution(ax, y: np.ndarray, label_names: dict, title: str):
    counts = {lbl: int((y == lbl).sum()) for lbl in sorted(label_names) if (y == lbl).any()}
    names  = [label_names[k] for k in counts]
    vals   = list(counts.values())
    bars   = ax.bar(names, vals, color='steelblue', edgecolor='white', linewidth=0.5)
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.set_ylabel('Window count')
    ax.tick_params(axis='x', rotation=35, labelsize=8)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                str(v), ha='center', va='bottom', fontsize=7)


def plot_sample_windows(pdf, X: np.ndarray, y: np.ndarray,
                        label_names: dict, dataset_name: str):
    """One figure per label: 3 example windows."""
    labels = [lbl for lbl in sorted(label_names) if (y == lbl).any()]
    t      = np.arange(WINDOW_LEN) / SAMPLE_HZ

    for lbl in labels:
        idx = np.where(y == lbl)[0]
        rng = np.random.default_rng(42)
        pick = idx[rng.choice(len(idx), size=min(3, len(idx)), replace=False)]

        fig, axes = plt.subplots(len(pick), 1, figsize=(12, 2.2 * len(pick)),
                                 sharex=True)
        if len(pick) == 1:
            axes = [axes]
        fig.suptitle(f'{dataset_name}  |  label {lbl}: {label_names[lbl]}',
                     fontsize=11, fontweight='bold')

        ch_colors = plt.cm.tab10(np.linspace(0, 1, 6))
        for ax, i in zip(axes, pick):
            for ch, (col, name) in enumerate(zip(ch_colors, ACCEL_COLS)):
                ax.plot(t, X[i, :, ch], color=col, lw=0.8, label=name)
            ax.set_ylabel('acc (g)', fontsize=8)
            ax.legend(ncol=6, fontsize=6, loc='upper right', framealpha=0.4)

        axes[-1].set_xlabel('time (s)', fontsize=9)
        _tight(fig)
        pdf.savefig(fig)
        plt.close(fig)


def plot_rolling_stats(pdf, fpath: str, keep_labels: set,
                       label_names: dict, dataset_name: str,
                       window_sec: float = 5.12):
    """Rolling mean and std on one subject's continuous recording."""
    df = load_subject(fpath, keep_labels)
    if len(df) < WINDOW_LEN * 2:
        return

    roll_w = int(window_sec * SAMPLE_HZ)
    fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True)
    fig.suptitle(f'{dataset_name}: rolling statistics — {os.path.basename(fpath)}',
                 fontsize=11, fontweight='bold')

    t = np.arange(len(df)) / SAMPLE_HZ

    # Raw signal (back_x as representative channel)
    ax = axes[0]
    ax.plot(t, df['back_x'].values, lw=0.4, color='steelblue', label='back_x')
    ax.set_ylabel('acc (g)', fontsize=9)
    ax.legend(fontsize=8)
    _annotate_labels(ax, df['label'].values, t, label_names)

    # Rolling mean per channel
    ax = axes[1]
    ch_colors = plt.cm.tab10(np.linspace(0, 1, 6))
    for col_name, color in zip(ACCEL_COLS, ch_colors):
        rm = pd.Series(df[col_name].values).rolling(roll_w, center=True).mean()
        ax.plot(t, rm, lw=0.8, color=color, label=col_name)
    ax.set_ylabel(f'rolling mean\n(w={window_sec}s)', fontsize=9)
    ax.legend(ncol=6, fontsize=6, loc='upper right', framealpha=0.4)
    ax.axhline(0, color='k', lw=0.4, ls='--')

    # Rolling std per channel
    ax = axes[2]
    for col_name, color in zip(ACCEL_COLS, ch_colors):
        rs = pd.Series(df[col_name].values).rolling(roll_w, center=True).std()
        ax.plot(t, rs, lw=0.8, color=color, label=col_name)
    ax.set_ylabel(f'rolling std\n(w={window_sec}s)', fontsize=9)
    ax.set_xlabel('time (s)', fontsize=9)
    ax.legend(ncol=6, fontsize=6, loc='upper right', framealpha=0.4)

    _tight(fig)
    pdf.savefig(fig)
    plt.close(fig)


def _annotate_labels(ax, labels_arr, t, label_names):
    """Shade background by activity label."""
    cmap   = plt.cm.Set3(np.linspace(0, 1, 12))
    unique = sorted(set(labels_arr))
    colors = {lbl: cmap[i % 12] for i, lbl in enumerate(unique)}
    runs   = []
    prev   = labels_arr[0]
    start  = 0
    for i, lbl in enumerate(labels_arr):
        if lbl != prev:
            runs.append((start, i - 1, prev))
            start = i
            prev  = lbl
    runs.append((start, len(labels_arr) - 1, prev))
    for s, e, lbl in runs:
        ax.axvspan(t[s], t[e], alpha=0.15, color=colors[lbl],
                   label=label_names.get(lbl, str(lbl)))


def plot_window_distributions(pdf, X: np.ndarray, y: np.ndarray,
                              label_names: dict, dataset_name: str):
    """Box plots of per-window mean and std, split by label and channel."""
    labels = [lbl for lbl in sorted(label_names) if (y == lbl).any()]
    label_short = [label_names[l][:8] for l in labels]

    for stat_name, fn in [('per-window mean', np.mean), ('per-window std', np.std)]:
        fig, axes = plt.subplots(2, 3, figsize=(14, 7), sharey=False)
        fig.suptitle(f'{dataset_name}: {stat_name} distribution by activity',
                     fontsize=11, fontweight='bold')
        for ch, ax in enumerate(axes.flat):
            data_by_label = [fn(X[y == lbl, :, ch], axis=1) for lbl in labels]
            bp = ax.boxplot(data_by_label, patch_artist=True, showfliers=False,
                            medianprops={'color': 'red', 'lw': 1.5})
            colors = plt.cm.Set2(np.linspace(0, 1, len(labels)))
            for patch, col in zip(bp['boxes'], colors):
                patch.set_facecolor(col)
                patch.set_alpha(0.7)
            ax.set_title(ACCEL_COLS[ch], fontsize=9)
            ax.set_xticks(range(1, len(labels) + 1))
            ax.set_xticklabels(label_short, rotation=35, fontsize=7)
            ax.set_ylabel(stat_name, fontsize=8)
            ax.axhline(0, color='k', lw=0.4, ls='--')
        _tight(fig)
        pdf.savefig(fig)
        plt.close(fig)


def plot_stationarity_heatmaps(pdf, adf_frac: dict, kpss_frac: dict,
                               label_names: dict, dataset_name: str):
    """Heatmap: fraction of stationary / non-stationary windows per channel × label.

    ADF: high value → most windows stationary (good).
    KPSS: high value → most windows non-stationary (bad).
    """
    labels  = sorted(adf_frac.keys())
    lnames  = [f'{label_names[l]}\n({l})' for l in labels]

    adf_mat  = np.array([adf_frac[l]  for l in labels]).T   # [6, n_labels]
    kpss_mat = np.array([kpss_frac[l] for l in labels]).T

    fig, axes = plt.subplots(1, 2, figsize=(max(10, len(labels) * 1.2 + 3), 5))
    fig.suptitle(
        f'{dataset_name}: stationarity test results  (α=0.05, per window)\n'
        'ADF: fraction where unit root rejected → stationary\n'
        'KPSS: fraction where stationarity rejected → non-stationary',
        fontsize=10, fontweight='bold'
    )

    for ax, mat, title, cmap in [
        (axes[0], adf_mat,  'ADF  (↑ = more stationary)',  'RdYlGn'),
        (axes[1], kpss_mat, 'KPSS (↑ = more non-stationary)', 'RdYlGn_r'),
    ]:
        im = ax.imshow(mat, aspect='auto', vmin=0, vmax=1, cmap=cmap)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(lnames, fontsize=7)
        ax.set_yticks(range(6))
        ax.set_yticklabels(ACCEL_COLS, fontsize=8)
        ax.set_title(title, fontsize=9, fontweight='bold')
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                v = mat[i, j]
                ax.text(j, i, f'{v:.2f}', ha='center', va='center', fontsize=7,
                        color='white' if (v > 0.7 or v < 0.3) else 'black')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    _tight(fig)
    pdf.savefig(fig)
    plt.close(fig)


def plot_acf_by_label(pdf, X: np.ndarray, y: np.ndarray,
                      label_names: dict, dataset_name: str,
                      max_lag: int = 50, n_examples: int = 20):
    """Mean ACF across windows per label for back_x (channel 0).

    A slowly decaying ACF → autocorrelation → tendency toward non-stationarity.
    """
    from statsmodels.tsa.stattools import acf

    labels = [lbl for lbl in sorted(label_names) if (y == lbl).any()]
    n_cols = min(4, len(labels))
    n_rows = int(np.ceil(len(labels) / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(n_cols * 3.5, n_rows * 2.8),
                             sharex=True, sharey=True)
    axes = np.array(axes).reshape(-1)
    fig.suptitle(f'{dataset_name}: mean ACF of back_x by activity\n'
                 '(shaded = mean ± std across windows; dashed = 95% CI)',
                 fontsize=10, fontweight='bold')

    lags = np.arange(max_lag + 1)
    conf = 1.96 / np.sqrt(WINDOW_LEN)   # approx 95% CI for white noise

    for ax, lbl in zip(axes, labels):
        idx = np.where(y == lbl)[0]
        rng = np.random.default_rng(0)
        pick = idx[rng.choice(len(idx), size=min(n_examples, len(idx)), replace=False)]
        acfs = np.stack([acf(X[i, :, 0], nlags=max_lag, fft=True) for i in pick])
        mean_acf = acfs.mean(axis=0)
        std_acf  = acfs.std(axis=0)

        ax.bar(lags, mean_acf, width=0.8, color='steelblue', alpha=0.7)
        ax.fill_between(lags, mean_acf - std_acf, mean_acf + std_acf,
                        alpha=0.3, color='steelblue')
        ax.axhline(conf,  color='r', lw=0.8, ls='--')
        ax.axhline(-conf, color='r', lw=0.8, ls='--')
        ax.axhline(0, color='k', lw=0.4)
        ax.set_title(f'{label_names[lbl]} ({lbl})\nn={len(pick)}', fontsize=8)
        ax.set_ylim(-0.5, 1.05)

    for ax in axes[len(labels):]:
        ax.set_visible(False)

    axes[0].set_ylabel('ACF', fontsize=8)
    fig.text(0.5, 0.01, 'lag (samples)', ha='center', fontsize=9)
    _tight(fig)
    pdf.savefig(fig)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_dataset(name: str, subject_file: str | None = None,
                max_windows_for_stat: int = 200):
    data_dir, label_names, keep_labels = DATASETS[name]

    if not os.path.isdir(data_dir):
        print(f'[{name}] Directory not found: {data_dir}  — skipping.')
        return

    files = sorted(glob.glob(os.path.join(data_dir, '*.csv')))
    if not files:
        print(f'[{name}] No CSV files found in {data_dir}  — skipping.')
        return

    subject = subject_file or files[0]

    print(f'\n[{name}] Loading all windows (may take a moment) …')
    X, y = load_all_windows(data_dir, keep_labels)
    print(f'[{name}] {len(X)} windows, {len(np.unique(y))} classes')

    # Sub-sample for stationarity tests to keep runtime reasonable
    rng   = np.random.default_rng(0)
    if len(X) > max_windows_for_stat:
        idx   = rng.choice(len(X), max_windows_for_stat, replace=False)
        X_st  = X[idx]
        y_st  = y[idx]
    else:
        X_st, y_st = X, y

    print(f'[{name}] Running ADF + KPSS tests on {len(X_st)} windows …')
    adf_frac, kpss_frac = stationarity_summary(X_st, y_st, label_names)

    out_pdf = f'explore_{name}.pdf'
    print(f'[{name}] Writing {out_pdf} …')

    with PdfPages(out_pdf) as pdf:
        # Page 1: class distribution
        fig, ax = plt.subplots(figsize=(10, 4))
        plot_class_distribution(ax, y, label_names,
                                f'{name.upper()} — class distribution')
        _tight(fig)
        pdf.savefig(fig)
        plt.close(fig)

        # Pages 2+: sample windows per activity
        plot_sample_windows(pdf, X, y, label_names, name.upper())

        # Rolling stats on one subject
        plot_rolling_stats(pdf, subject, keep_labels, label_names, name.upper())

        # Window mean / std distributions
        plot_window_distributions(pdf, X, y, label_names, name.upper())

        # ACF by label
        plot_acf_by_label(pdf, X, y, label_names, name.upper())

        # Stationarity heatmaps
        plot_stationarity_heatmaps(pdf, adf_frac, kpss_frac,
                                   label_names, name.upper())

    print(f'[{name}] Done → {out_pdf}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset',  choices=['harth', 'har70plus', 'both'],
                        default='both')
    parser.add_argument('--subject',  default=None,
                        help='Path to a specific subject CSV for rolling-stats plot')
    parser.add_argument('--max_stat', type=int, default=200,
                        help='Max windows per dataset to run ADF/KPSS on '
                             '(for speed; full dataset used for visuals)')
    args = parser.parse_args()

    targets = list(DATASETS) if args.dataset == 'both' else [args.dataset]
    for ds in targets:
        run_dataset(ds, subject_file=args.subject, max_windows_for_stat=args.max_stat)

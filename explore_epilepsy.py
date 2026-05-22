"""explore_epilepsy.py — visual and statistical exploration of Epilepsy / SleepEEG.

Produces a PDF per dataset with:
  1. Class distribution (window counts).
  2. Sample windows per class.
  3. Shuffling example: original vs. time-shuffled window — side-by-side raw
     signal and ACF — to demonstrate (near-)stationarity vs. the non-stationary
     character of HARTH/HAR70plus signals.
  4. Per-window mean and std distributions, split by class.
  5. ACF by class (mean ± std across windows).
  6. ADF + KPSS stationarity heatmaps.

Usage
-----
  python explore_epilepsy.py                         # both datasets
  python explore_epilepsy.py --dataset epilepsy
  python explore_epilepsy.py --dataset sleepeeg

The script reads preprocessed pickle files from preprocessed_data/.
Data format: list of 12 elements
  [0] X_train [N, C, L], [3] y_train, [4] X_val, [7] y_val, [8] X_test, [11] y_test

Dependencies: numpy, pandas, matplotlib, scipy, statsmodels
"""

import argparse
import os
import pickle
import warnings

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from statsmodels.tsa.stattools import adfuller, kpss, acf as sm_acf

# ---------------------------------------------------------------------------
# Dataset configuration
# ---------------------------------------------------------------------------

WINDOW_LEN  = 256
SAMPLE_HZ   = 100   # Epilepsy EEG dataset (Bonn) is 173.6 Hz, but stored at 100 Hz
                     # SleepEEG (SleepEDF Cassette) is also 100 Hz
EEG_COL     = 'EEG'

EPILEPSY_LABEL_NAMES = {
    0: 'non-seizure',
    1: 'seizure',
}
SLEEPEEG_LABEL_NAMES = {
    0: 'Wake',
    1: 'N1',
    2: 'N2',
    3: 'N3',
    4: 'REM',
}

DATASETS = {
    'epilepsy': ('preprocessed_data/_DA_Epilepsy_256_00.pkl', EPILEPSY_LABEL_NAMES),
    'sleepeeg': ('preprocessed_data/_DA_SleepEEG_256_00.pkl', SLEEPEEG_LABEL_NAMES),
}

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_dataset(pkl_path: str):
    """Load all splits and return concatenated (X [N, L, 1], y [N])."""
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    # indices: 0=X_train, 3=y_train, 4=X_val, 7=y_val, 8=X_test, 11=y_test
    xs, ys = [], []
    for xi, yi in [(0, 3), (4, 7), (8, 11)]:
        if data[xi] is not None and data[yi] is not None:
            X = data[xi]   # [N, C, L]
            y = data[yi]   # [N]
            xs.append(X.transpose(0, 2, 1))   # → [N, L, C]
            ys.append(y)
    return np.concatenate(xs), np.concatenate(ys)

# ---------------------------------------------------------------------------
# Stationarity tests
# ---------------------------------------------------------------------------

def adf_pvalue(series: np.ndarray) -> float:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            result = adfuller(series, autolag='AIC', maxlag=10)
        return float(result[1])
    except Exception:
        return float('nan')


def kpss_pvalue(series: np.ndarray) -> float:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            result = kpss(series, regression='c', nlags='auto')
        return float(result[1])
    except Exception:
        return float('nan')


def stationarity_summary(X: np.ndarray, y: np.ndarray,
                         label_names: dict, alpha: float = 0.05):
    labels    = sorted(label_names)
    adf_frac  = {}
    kpss_frac = {}
    for lbl in labels:
        X_lbl = X[y == lbl]
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
# Plotting
# ---------------------------------------------------------------------------

def _tight(fig):
    fig.tight_layout()


def plot_class_distribution(ax, y: np.ndarray, label_names: dict, title: str):
    counts = {lbl: int((y == lbl).sum()) for lbl in sorted(label_names) if (y == lbl).any()}
    names  = [label_names[k] for k in counts]
    vals   = list(counts.values())
    bars   = ax.bar(names, vals, color='mediumseagreen', edgecolor='white', linewidth=0.5)
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.set_ylabel('Window count')
    ax.tick_params(axis='x', rotation=20, labelsize=9)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                str(v), ha='center', va='bottom', fontsize=8)


def plot_sample_windows(pdf, X: np.ndarray, y: np.ndarray,
                        label_names: dict, dataset_name: str):
    """3 example windows per class."""
    t   = np.arange(WINDOW_LEN) / SAMPLE_HZ
    for lbl in sorted(label_names):
        idx = np.where(y == lbl)[0]
        if len(idx) == 0:
            continue
        rng  = np.random.default_rng(42)
        pick = idx[rng.choice(len(idx), size=min(3, len(idx)), replace=False)]

        fig, axes = plt.subplots(len(pick), 1, figsize=(12, 2.2 * len(pick)), sharex=True)
        if len(pick) == 1:
            axes = [axes]
        fig.suptitle(f'{dataset_name}  |  class {lbl}: {label_names[lbl]}',
                     fontsize=11, fontweight='bold')
        for ax, i in zip(axes, pick):
            ax.plot(t, X[i, :, 0], lw=0.8, color='steelblue')
            ax.set_ylabel('amplitude', fontsize=8)
        axes[-1].set_xlabel('time (s)', fontsize=9)
        _tight(fig)
        pdf.savefig(fig)
        plt.close(fig)


def plot_shuffling_example(pdf, X: np.ndarray, y: np.ndarray,
                           label_names: dict, dataset_name: str,
                           n_examples: int = 3, max_lag: int = 50):
    """For each class, show n_examples windows: original vs. time-shuffled.

    Left column: raw signal.  Right column: ACF.
    A stationary signal looks nearly the same after shuffling (autocorrelation
    collapses to near-zero).  Non-stationary signals (HARTH walking/running)
    show clear structure that shuffling destroys.
    """
    rng  = np.random.default_rng(0)
    conf = 1.96 / np.sqrt(WINDOW_LEN)
    t    = np.arange(WINDOW_LEN) / SAMPLE_HZ
    lags = np.arange(max_lag + 1)

    for lbl in sorted(label_names):
        idx = np.where(y == lbl)[0]
        if len(idx) == 0:
            continue
        pick = idx[rng.choice(len(idx), size=min(n_examples, len(idx)), replace=False)]

        n_rows = len(pick)
        fig, axes = plt.subplots(n_rows, 4,
                                 figsize=(14, 2.5 * n_rows),
                                 gridspec_kw={'width_ratios': [2, 1, 2, 1]})
        if n_rows == 1:
            axes = axes[np.newaxis, :]

        fig.suptitle(
            f'{dataset_name}  |  class {lbl}: {label_names[lbl]}\n'
            'Columns: original signal | original ACF | shuffled signal | shuffled ACF\n'
            '(If stationary, original ≈ shuffled in both signal and ACF)',
            fontsize=9, fontweight='bold'
        )

        for row, i in enumerate(pick):
            sig      = X[i, :, 0].copy()
            sig_shuf = rng.permutation(sig)

            acf_orig = sm_acf(sig,      nlags=max_lag, fft=True)
            acf_shuf = sm_acf(sig_shuf, nlags=max_lag, fft=True)

            # Original signal
            axes[row, 0].plot(t, sig, lw=0.7, color='steelblue')
            axes[row, 0].set_ylabel('amp', fontsize=7)
            if row == 0:
                axes[row, 0].set_title('Original', fontsize=8, fontweight='bold')

            # Original ACF
            axes[row, 1].bar(lags, acf_orig, width=0.8, color='steelblue', alpha=0.7)
            axes[row, 1].axhline( conf, color='r', lw=0.8, ls='--')
            axes[row, 1].axhline(-conf, color='r', lw=0.8, ls='--')
            axes[row, 1].axhline(0, color='k', lw=0.4)
            axes[row, 1].set_ylim(-0.6, 1.05)
            if row == 0:
                axes[row, 1].set_title('ACF (orig)', fontsize=8, fontweight='bold')

            # Shuffled signal
            axes[row, 2].plot(t, sig_shuf, lw=0.7, color='darkorange')
            if row == 0:
                axes[row, 2].set_title('Shuffled', fontsize=8, fontweight='bold')

            # Shuffled ACF
            axes[row, 3].bar(lags, acf_shuf, width=0.8, color='darkorange', alpha=0.7)
            axes[row, 3].axhline( conf, color='r', lw=0.8, ls='--')
            axes[row, 3].axhline(-conf, color='r', lw=0.8, ls='--')
            axes[row, 3].axhline(0, color='k', lw=0.4)
            axes[row, 3].set_ylim(-0.6, 1.05)
            if row == 0:
                axes[row, 3].set_title('ACF (shuffled)', fontsize=8, fontweight='bold')

        for ax in axes[-1, [0, 2]]:
            ax.set_xlabel('time (s)', fontsize=7)
        for ax in axes[-1, [1, 3]]:
            ax.set_xlabel('lag', fontsize=7)

        _tight(fig)
        pdf.savefig(fig)
        plt.close(fig)


def plot_window_distributions(pdf, X: np.ndarray, y: np.ndarray,
                              label_names: dict, dataset_name: str):
    """Box plots of per-window mean and std, split by class."""
    labels      = [lbl for lbl in sorted(label_names) if (y == lbl).any()]
    label_names_short = [label_names[l][:10] for l in labels]

    for stat_name, fn in [('per-window mean', np.mean), ('per-window std', np.std)]:
        fig, ax = plt.subplots(figsize=(max(6, len(labels) * 1.5 + 2), 4))
        fig.suptitle(f'{dataset_name}: {stat_name} distribution by class',
                     fontsize=11, fontweight='bold')
        data_by_label = [fn(X[y == lbl, :, 0], axis=1) for lbl in labels]
        bp = ax.boxplot(data_by_label, patch_artist=True, showfliers=False,
                        medianprops={'color': 'red', 'lw': 1.5})
        colors = plt.cm.Set2(np.linspace(0, 1, len(labels)))
        for patch, col in zip(bp['boxes'], colors):
            patch.set_facecolor(col)
            patch.set_alpha(0.7)
        ax.set_xticks(range(1, len(labels) + 1))
        ax.set_xticklabels(label_names_short, rotation=20, fontsize=9)
        ax.set_ylabel(stat_name, fontsize=9)
        ax.axhline(0, color='k', lw=0.4, ls='--')
        _tight(fig)
        pdf.savefig(fig)
        plt.close(fig)


def plot_acf_by_label(pdf, X: np.ndarray, y: np.ndarray,
                      label_names: dict, dataset_name: str,
                      max_lag: int = 50, n_examples: int = 30):
    """Mean ACF ± std across windows per class."""
    labels  = [lbl for lbl in sorted(label_names) if (y == lbl).any()]
    n_cols  = min(3, len(labels))
    n_rows  = int(np.ceil(len(labels) / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(n_cols * 4, n_rows * 3),
                             sharex=True, sharey=True)
    axes = np.array(axes).reshape(-1)
    fig.suptitle(f'{dataset_name}: mean ACF by class\n'
                 '(shaded = mean ± std across windows; dashed = 95% CI for white noise)',
                 fontsize=10, fontweight='bold')

    lags = np.arange(max_lag + 1)
    conf = 1.96 / np.sqrt(WINDOW_LEN)

    for ax, lbl in zip(axes, labels):
        idx  = np.where(y == lbl)[0]
        rng  = np.random.default_rng(0)
        pick = idx[rng.choice(len(idx), size=min(n_examples, len(idx)), replace=False)]
        acfs = np.stack([sm_acf(X[i, :, 0], nlags=max_lag, fft=True) for i in pick])
        mean_acf = acfs.mean(axis=0)
        std_acf  = acfs.std(axis=0)

        ax.bar(lags, mean_acf, width=0.8, color='mediumseagreen', alpha=0.7)
        ax.fill_between(lags, mean_acf - std_acf, mean_acf + std_acf,
                        alpha=0.3, color='mediumseagreen')
        ax.axhline( conf, color='r', lw=0.8, ls='--')
        ax.axhline(-conf, color='r', lw=0.8, ls='--')
        ax.axhline(0, color='k', lw=0.4)
        ax.set_title(f'{label_names[lbl]} ({lbl})\nn={len(pick)}', fontsize=9)
        ax.set_ylim(-0.5, 1.05)

    for ax in axes[len(labels):]:
        ax.set_visible(False)

    axes[0].set_ylabel('ACF', fontsize=8)
    fig.text(0.5, 0.01, 'lag (samples)', ha='center', fontsize=9)
    _tight(fig)
    pdf.savefig(fig)
    plt.close(fig)


def plot_stationarity_heatmaps(pdf, adf_frac: dict, kpss_frac: dict,
                               label_names: dict, dataset_name: str):
    labels  = sorted(adf_frac.keys())
    lnames  = [f'{label_names[l]}\n({l})' for l in labels]

    adf_mat  = np.array([adf_frac[l]  for l in labels]).T   # [n_ch, n_labels]
    kpss_mat = np.array([kpss_frac[l] for l in labels]).T
    ch_names = ['EEG']

    fig, axes = plt.subplots(1, 2, figsize=(max(8, len(labels) * 1.6 + 3), 3.5))
    fig.suptitle(
        f'{dataset_name}: stationarity test results  (α=0.05, per window)\n'
        'ADF: fraction where unit root rejected → stationary\n'
        'KPSS: fraction where stationarity rejected → non-stationary',
        fontsize=10, fontweight='bold'
    )

    for ax, mat, title, cmap in [
        (axes[0], adf_mat,  'ADF  (↑ = more stationary)',        'RdYlGn'),
        (axes[1], kpss_mat, 'KPSS (↑ = more non-stationary)',    'RdYlGn_r'),
    ]:
        im = ax.imshow(mat, aspect='auto', vmin=0, vmax=1, cmap=cmap)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(lnames, fontsize=8)
        ax.set_yticks(range(len(ch_names)))
        ax.set_yticklabels(ch_names, fontsize=9)
        ax.set_title(title, fontsize=9, fontweight='bold')
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                v = mat[i, j]
                ax.text(j, i, f'{v:.2f}', ha='center', va='center', fontsize=10,
                        color='white' if (v > 0.7 or v < 0.3) else 'black')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    _tight(fig)
    pdf.savefig(fig)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_dataset(name: str, max_windows_for_stat: int = 300):
    pkl_path, label_names = DATASETS[name]

    if not os.path.exists(pkl_path):
        print(f'[{name}] Pickle not found: {pkl_path}  — skipping.')
        return

    print(f'\n[{name}] Loading data …')
    X, y = load_dataset(pkl_path)
    print(f'[{name}] {len(X)} windows, {len(np.unique(y))} classes, '
          f'X shape {X.shape}')
    for lbl in sorted(label_names):
        n = (y == lbl).sum()
        if n:
            print(f'         class {lbl} ({label_names[lbl]}): {n} windows')

    rng = np.random.default_rng(0)
    if len(X) > max_windows_for_stat:
        idx  = rng.choice(len(X), max_windows_for_stat, replace=False)
        X_st = X[idx]
        y_st = y[idx]
    else:
        X_st, y_st = X, y

    print(f'[{name}] Running ADF + KPSS on {len(X_st)} windows …')
    adf_frac, kpss_frac = stationarity_summary(X_st, y_st, label_names)

    out_pdf = f'explore_{name}.pdf'
    print(f'[{name}] Writing {out_pdf} …')

    with PdfPages(out_pdf) as pdf:
        # Page 1: class distribution
        fig, ax = plt.subplots(figsize=(8, 4))
        plot_class_distribution(ax, y, label_names,
                                f'{name.upper()} — class distribution')
        _tight(fig)
        pdf.savefig(fig)
        plt.close(fig)

        # Sample windows per class
        plot_sample_windows(pdf, X, y, label_names, name.upper())

        # Shuffling comparison (key stationarity visual)
        plot_shuffling_example(pdf, X, y, label_names, name.upper())

        # Window mean / std distributions
        plot_window_distributions(pdf, X, y, label_names, name.upper())

        # ACF by class
        plot_acf_by_label(pdf, X, y, label_names, name.upper())

        # Stationarity heatmaps
        plot_stationarity_heatmaps(pdf, adf_frac, kpss_frac,
                                   label_names, name.upper())

    print(f'[{name}] Done → {out_pdf}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=['epilepsy', 'sleepeeg', 'both'],
                        default='both')
    parser.add_argument('--max_stat', type=int, default=300,
                        help='Max windows per dataset for ADF/KPSS (speed)')
    args = parser.parse_args()

    targets = list(DATASETS) if args.dataset == 'both' else [args.dataset]
    for ds in targets:
        run_dataset(ds, max_windows_for_stat=args.max_stat)

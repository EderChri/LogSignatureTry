"""explore_capture24.py — visual and statistical exploration of capture24.

Produces a PDF with:
  1. Category distribution (window counts) — sleep / sedentary / light / MVPA,
     derived from the MET value embedded in each row's `annotation` string
     (sleeping -> sleep; MET<1.5 -> sedentary; MET<3 -> light; else -> MVPA).
  2. Sample windows per category.
  3. Rolling mean / rolling std on a chunk of one subject's recording, with
     category shading.
  4. Per-window mean and std distributions, split by category.
  5. ADF + KPSS stationarity tests per channel and category.
  6. Mean ACF per category.

Usage
-----
  python explore_capture24.py                          # default: 5 subjects
  python explore_capture24.py --n_subjects 10
  python explore_capture24.py --subject data/capture24/P003.csv.gz
  python explore_capture24.py --max_windows_per_subject 5000

The script reads raw gzipped CSVs from data/capture24/ (P*.csv.gz, columns
time,x,y,z,annotation). No preprocessed pickle files are required. capture24
is used purely for unsupervised pretraining in this project (no labels are
used in training) — the category split here is only for data exploration.

Each subject's recording starts around 2am and runs ~24h, so the start of the
file is almost always 'sleep'. The script streams each file in chunks and
draws windows via reservoir sampling, giving an unbiased sample across the
*whole* recording instead of just its first minutes.

Dependencies: numpy, pandas, matplotlib, scipy, statsmodels
"""

import argparse
import glob
import os
import re
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from statsmodels.tsa.stattools import adfuller, kpss, acf as sm_acf

# ---------------------------------------------------------------------------
# Dataset configuration
# ---------------------------------------------------------------------------

ACCEL_COLS = ['x', 'y', 'z']
WINDOW_LEN = 256
STRIDE     = 128
SAMPLE_HZ  = 100   # capture24 (Axivity AX3) is sampled at 100 Hz
CHUNK_SIZE = 500_000   # rows per streamed chunk (~83 min at 100 Hz)

MET_RE = re.compile(r'MET\s*([\d.]+)')

CATEGORIES = ['sleep', 'sedentary', 'light', 'moderate-vigorous']
CATEGORY_COLOR = {
    'sleep':             '#4c72b0',
    'sedentary':         '#8c8c8c',
    'light':             '#55a868',
    'moderate-vigorous': '#c44e52',
}

# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def categorize(annotation) -> str:
    """Map a capture24 compendium annotation string to a coarse intensity category."""
    if not isinstance(annotation, str):
        return 'unknown'
    low = annotation.lower()
    if 'sleeping' in low:
        return 'sleep'
    m = MET_RE.search(annotation)
    if not m:
        return 'unknown'
    met = float(m.group(1))
    if met < 1.5:
        return 'sedentary'
    elif met < 3.0:
        return 'light'
    return 'moderate-vigorous'


def _extract_chunk_windows(values, cats, reservoir_X, reservoir_y, seen, cap, rng):
    """Reservoir-sample category-pure windows from one chunk into reservoir_X/y in place."""
    for start in range(0, len(values) - WINDOW_LEN + 1, STRIDE):
        end = start + WINDOW_LEN
        uq = set(cats[start:end])
        if len(uq) != 1:
            continue
        cat = next(iter(uq))
        if cat not in CATEGORIES:
            continue
        if seen < cap:
            reservoir_X[seen] = values[start:end]
            reservoir_y[seen] = cat
        else:
            j = rng.integers(0, seen + 1)
            if j < cap:
                reservoir_X[j] = values[start:end]
                reservoir_y[j] = cat
        seen += 1
    return seen


def process_subject(fpath: str, cap: int, rng: np.random.Generator,
                    rolling_rows: int | None = None):
    """Stream one subject's file in chunks.

    Returns (windows [n<=cap, L, 3] float32, categories [n] object, segment_df).
    Windows are drawn via reservoir sampling, giving an unbiased sample across
    the *entire* recording (the file starts mid-sleep, so naively taking the
    first rows would yield almost only the 'sleep' category).
    segment_df (only computed when rolling_rows is given) is the most
    category-diverse rolling_rows-long contiguous chunk-prefix found while
    streaming — used for the rolling-stats plot.
    """
    reservoir_X = np.empty((cap, WINDOW_LEN, 3), dtype=np.float32)
    reservoir_y = np.empty(cap, dtype=object)
    seen = 0
    best_segment, best_diversity = None, -1

    for chunk in pd.read_csv(fpath, compression='gzip',
                             usecols=ACCEL_COLS + ['annotation'],
                             chunksize=CHUNK_SIZE, low_memory=False):
        chunk = chunk.dropna(subset=ACCEL_COLS)
        if len(chunk) == 0:
            continue
        cats = chunk['annotation'].map(categorize).values
        values = chunk[ACCEL_COLS].values.astype(np.float32)

        seen = _extract_chunk_windows(values, cats, reservoir_X, reservoir_y, seen, cap, rng)

        if rolling_rows is not None:
            seg_len = min(rolling_rows, len(chunk))
            diversity = len(set(cats[:seg_len]) & set(CATEGORIES))
            if diversity > best_diversity:
                best_diversity = diversity
                seg = chunk.iloc[:seg_len][ACCEL_COLS].copy()
                seg['category'] = cats[:seg_len]
                best_segment = seg.reset_index(drop=True)
            if best_diversity == len(CATEGORIES):
                rolling_rows = None   # fully diverse segment found, stop searching

    n = min(seen, cap)
    return reservoir_X[:n], reservoir_y[:n], best_segment


def load_all_windows(data_dir: str, n_subjects: int, cap_per_subject: int,
                     rng: np.random.Generator,
                     rolling_subject: str | None = None, rolling_rows: int | None = None):
    """Pool reservoir-sampled windows from the first n_subjects participants.

    Returns (X, y, segment_df, segment_path) where segment_df is the most
    diverse rolling_rows-long stretch found for rolling_subject (or None).
    """
    files = sorted(glob.glob(os.path.join(data_dir, 'P*.csv.gz')))[:n_subjects]
    if not files:
        raise FileNotFoundError(f"No P*.csv.gz files found in '{data_dir}'")

    all_x, all_y = [], []
    segment_df, segment_path = None, None

    for fp in files:
        want_segment = rolling_rows is not None and fp == rolling_subject
        x, y, seg = process_subject(fp, cap_per_subject, rng,
                                    rolling_rows=rolling_rows if want_segment else None)
        print(f'  {os.path.basename(fp)}: {len(x)} windows (reservoir-sampled)')
        if len(x):
            all_x.append(x)
            all_y.append(y)
        if want_segment and seg is not None:
            segment_df, segment_path = seg, fp

    if rolling_subject is not None and rolling_subject not in files and segment_df is None:
        print(f'  Scanning {os.path.basename(rolling_subject)} for a representative segment …')
        _, _, seg = process_subject(rolling_subject, cap=1, rng=rng, rolling_rows=rolling_rows)
        if seg is not None:
            segment_df, segment_path = seg, rolling_subject

    if not all_x:
        raise RuntimeError('No windows extracted from capture24 data.')
    return np.concatenate(all_x), np.concatenate(all_y), segment_df, segment_path

# ---------------------------------------------------------------------------
# Stationarity tests
# ---------------------------------------------------------------------------

def adf_pvalue(series: np.ndarray) -> float:
    """ADF test p-value. Low p -> reject unit root -> stationary."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            result = adfuller(series, autolag='AIC', maxlag=10)
        return float(result[1])
    except Exception:
        return float('nan')


def kpss_pvalue(series: np.ndarray) -> float:
    """KPSS test p-value. Low p -> reject stationarity -> non-stationary."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            result = kpss(series, regression='c', nlags='auto')
        return float(result[1])
    except Exception:
        return float('nan')


def stationarity_summary(X: np.ndarray, y: np.ndarray, alpha: float = 0.05):
    """Per-channel, per-category ADF/KPSS rejection fractions."""
    adf_frac, kpss_frac = {}, {}
    for cat in CATEGORIES:
        X_cat = X[y == cat]
        if len(X_cat) == 0:
            continue
        n_ch   = X_cat.shape[2]
        adf_r  = np.zeros(n_ch)
        kpss_r = np.zeros(n_ch)
        for ch in range(n_ch):
            adf_ps  = [adf_pvalue(X_cat[i, :, ch])  for i in range(len(X_cat))]
            kpss_ps = [kpss_pvalue(X_cat[i, :, ch]) for i in range(len(X_cat))]
            adf_r[ch]  = np.nanmean(np.array(adf_ps)  < alpha)
            kpss_r[ch] = np.nanmean(np.array(kpss_ps) < alpha)
        adf_frac[cat]  = adf_r
        kpss_frac[cat] = kpss_r
    return adf_frac, kpss_frac

# ---------------------------------------------------------------------------
# Plotting functions
# ---------------------------------------------------------------------------

def _tight(fig):
    fig.tight_layout()


def plot_class_distribution(ax, y: np.ndarray, title: str):
    counts = {cat: int((y == cat).sum()) for cat in CATEGORIES if (y == cat).any()}
    names  = list(counts.keys())
    vals   = list(counts.values())
    colors = [CATEGORY_COLOR[c] for c in names]
    bars   = ax.bar(names, vals, color=colors, edgecolor='white', linewidth=0.5)
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.set_ylabel('Window count')
    ax.tick_params(axis='x', rotation=15, labelsize=9)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                str(v), ha='center', va='bottom', fontsize=8)


def plot_sample_windows(pdf, X: np.ndarray, y: np.ndarray, dataset_name: str):
    """One figure per category: 3 example windows."""
    t = np.arange(WINDOW_LEN) / SAMPLE_HZ
    ch_colors = ['tab:blue', 'tab:orange', 'tab:green']

    for cat in CATEGORIES:
        idx = np.where(y == cat)[0]
        if len(idx) == 0:
            continue
        rng  = np.random.default_rng(42)
        pick = idx[rng.choice(len(idx), size=min(3, len(idx)), replace=False)]

        fig, axes = plt.subplots(len(pick), 1, figsize=(12, 2.2 * len(pick)), sharex=True)
        if len(pick) == 1:
            axes = [axes]
        fig.suptitle(f'{dataset_name}  |  category: {cat}', fontsize=11, fontweight='bold')

        for ax, i in zip(axes, pick):
            for ch, (col, name) in enumerate(zip(ch_colors, ACCEL_COLS)):
                ax.plot(t, X[i, :, ch], color=col, lw=0.8, label=name)
            ax.set_ylabel('acc (g)', fontsize=8)
            ax.legend(ncol=3, fontsize=7, loc='upper right', framealpha=0.4)

        axes[-1].set_xlabel('time (s)', fontsize=9)
        _tight(fig)
        pdf.savefig(fig)
        plt.close(fig)


def _annotate_categories(ax, cats_arr, t):
    """Shade background by intensity category."""
    runs = []
    prev, start = cats_arr[0], 0
    for i, c in enumerate(cats_arr):
        if c != prev:
            runs.append((start, i - 1, prev))
            start, prev = i, c
    runs.append((start, len(cats_arr) - 1, prev))
    seen = set()
    for s, e, c in runs:
        color = CATEGORY_COLOR.get(c, '#dddddd')
        label = c if c not in seen else None
        seen.add(c)
        ax.axvspan(t[s], t[e], alpha=0.15, color=color, label=label)


def plot_rolling_stats(pdf, segment_df, segment_path: str | None, dataset_name: str,
                       window_sec: float = 5.12):
    """Rolling mean and std on the most category-diverse stretch found for one subject."""
    if segment_df is None or len(segment_df) < WINDOW_LEN * 2:
        print('[capture24] Skipping rolling-stats plot: no representative segment found.')
        return
    df = segment_df

    roll_w = int(window_sec * SAMPLE_HZ)
    fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True)
    minutes = len(df) / SAMPLE_HZ / 60
    fig.suptitle(f'{dataset_name}: rolling statistics — {os.path.basename(segment_path)} '
                f'(most diverse {minutes:.0f}-min stretch found)',
                fontsize=11, fontweight='bold')

    t = np.arange(len(df)) / SAMPLE_HZ

    # Raw signal (x as representative channel)
    ax = axes[0]
    ax.plot(t, df['x'].values, lw=0.4, color='tab:blue', label='x')
    ax.set_ylabel('acc (g)', fontsize=9)
    _annotate_categories(ax, df['category'].values, t)
    ax.legend(ncol=5, fontsize=7, loc='upper right', framealpha=0.4)

    # Rolling mean per channel
    ax = axes[1]
    for col_name, color in zip(ACCEL_COLS, ['tab:blue', 'tab:orange', 'tab:green']):
        rm = pd.Series(df[col_name].values).rolling(roll_w, center=True).mean()
        ax.plot(t, rm, lw=0.8, color=color, label=col_name)
    ax.set_ylabel(f'rolling mean\n(w={window_sec}s)', fontsize=9)
    ax.legend(ncol=3, fontsize=7, loc='upper right', framealpha=0.4)
    ax.axhline(0, color='k', lw=0.4, ls='--')

    # Rolling std per channel
    ax = axes[2]
    for col_name, color in zip(ACCEL_COLS, ['tab:blue', 'tab:orange', 'tab:green']):
        rs = pd.Series(df[col_name].values).rolling(roll_w, center=True).std()
        ax.plot(t, rs, lw=0.8, color=color, label=col_name)
    ax.set_ylabel(f'rolling std\n(w={window_sec}s)', fontsize=9)
    ax.set_xlabel('time (s)', fontsize=9)
    ax.legend(ncol=3, fontsize=7, loc='upper right', framealpha=0.4)

    _tight(fig)
    pdf.savefig(fig)
    plt.close(fig)


def plot_window_distributions(pdf, X: np.ndarray, y: np.ndarray, dataset_name: str):
    """Box plots of per-window mean and std, split by category and channel."""
    cats = [c for c in CATEGORIES if (y == c).any()]

    for stat_name, fn in [('per-window mean', np.mean), ('per-window std', np.std)]:
        fig, axes = plt.subplots(1, 3, figsize=(13, 4.5), sharey=False)
        fig.suptitle(f'{dataset_name}: {stat_name} distribution by category',
                     fontsize=11, fontweight='bold')
        for ch, ax in enumerate(axes):
            data_by_cat = [fn(X[y == c, :, ch], axis=1) for c in cats]
            bp = ax.boxplot(data_by_cat, patch_artist=True, showfliers=False,
                            medianprops={'color': 'red', 'lw': 1.5})
            for patch, c in zip(bp['boxes'], cats):
                patch.set_facecolor(CATEGORY_COLOR[c])
                patch.set_alpha(0.7)
            ax.set_title(ACCEL_COLS[ch], fontsize=9)
            ax.set_xticks(range(1, len(cats) + 1))
            ax.set_xticklabels(cats, rotation=20, fontsize=8)
            ax.set_ylabel(stat_name, fontsize=8)
            ax.axhline(0, color='k', lw=0.4, ls='--')
        _tight(fig)
        pdf.savefig(fig)
        plt.close(fig)


def plot_acf_by_category(pdf, X: np.ndarray, y: np.ndarray, dataset_name: str,
                         max_lag: int = 50, n_examples: int = 30):
    """Mean ACF across windows per category for the x channel."""
    cats   = [c for c in CATEGORIES if (y == c).any()]
    n_cols = min(4, len(cats))
    n_rows = int(np.ceil(len(cats) / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3.5, n_rows * 2.8),
                             sharex=True, sharey=True)
    axes = np.array(axes).reshape(-1)
    fig.suptitle(f'{dataset_name}: mean ACF of x-axis by category\n'
                 '(shaded = mean ± std across windows; dashed = 95% CI)',
                 fontsize=10, fontweight='bold')

    lags = np.arange(max_lag + 1)
    conf = 1.96 / np.sqrt(WINDOW_LEN)

    for ax, cat in zip(axes, cats):
        idx  = np.where(y == cat)[0]
        rng  = np.random.default_rng(0)
        pick = idx[rng.choice(len(idx), size=min(n_examples, len(idx)), replace=False)]
        acfs = np.stack([sm_acf(X[i, :, 0], nlags=max_lag, fft=True) for i in pick])
        mean_acf, std_acf = acfs.mean(axis=0), acfs.std(axis=0)

        color = CATEGORY_COLOR[cat]
        ax.bar(lags, mean_acf, width=0.8, color=color, alpha=0.7)
        ax.fill_between(lags, mean_acf - std_acf, mean_acf + std_acf, alpha=0.3, color=color)
        ax.axhline(conf, color='r', lw=0.8, ls='--')
        ax.axhline(-conf, color='r', lw=0.8, ls='--')
        ax.axhline(0, color='k', lw=0.4)
        ax.set_title(f'{cat}\nn={len(pick)}', fontsize=8)
        ax.set_ylim(-0.5, 1.05)

    for ax in axes[len(cats):]:
        ax.set_visible(False)

    axes[0].set_ylabel('ACF', fontsize=8)
    fig.text(0.5, 0.01, 'lag (samples)', ha='center', fontsize=9)
    _tight(fig)
    pdf.savefig(fig)
    plt.close(fig)


def plot_stationarity_heatmaps(pdf, adf_frac: dict, kpss_frac: dict, dataset_name: str):
    """Heatmap: fraction of stationary / non-stationary windows per channel x category."""
    cats = list(adf_frac.keys())
    adf_mat  = np.array([adf_frac[c]  for c in cats]).T   # [3, n_cats]
    kpss_mat = np.array([kpss_frac[c] for c in cats]).T

    fig, axes = plt.subplots(1, 2, figsize=(max(8, len(cats) * 1.6 + 3), 3.5))
    fig.suptitle(
        f'{dataset_name}: stationarity test results  (α=0.05, per window)\n'
        'ADF: fraction where unit root rejected → stationary\n'
        'KPSS: fraction where stationarity rejected → non-stationary',
        fontsize=10, fontweight='bold'
    )
    for ax, mat, title, cmap in [
        (axes[0], adf_mat,  'ADF  (↑ = more stationary)',     'RdYlGn'),
        (axes[1], kpss_mat, 'KPSS (↑ = more non-stationary)', 'RdYlGn_r'),
    ]:
        im = ax.imshow(mat, aspect='auto', vmin=0, vmax=1, cmap=cmap)
        ax.set_xticks(range(len(cats)))
        ax.set_xticklabels(cats, fontsize=8, rotation=15)
        ax.set_yticks(range(3))
        ax.set_yticklabels(ACCEL_COLS, fontsize=9)
        ax.set_title(title, fontsize=9, fontweight='bold')
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                v = mat[i, j]
                ax.text(j, i, f'{v:.2f}', ha='center', va='center', fontsize=9,
                        color='white' if (v > 0.7 or v < 0.3) else 'black')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    _tight(fig)
    pdf.savefig(fig)
    plt.close(fig)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(data_dir: str, n_subjects: int, max_windows_per_subject: int,
       subject_file: str | None, rolling_minutes: float, max_windows_for_stat: int,
       seed: int):
    if not os.path.isdir(data_dir):
        print(f'Directory not found: {data_dir}  — skipping.')
        return

    files = sorted(glob.glob(os.path.join(data_dir, 'P*.csv.gz')))
    if not files:
        print(f'No P*.csv.gz files found in {data_dir}  — skipping.')
        return

    rolling_subject = subject_file or files[0]
    rolling_rows = int(rolling_minutes * 60 * SAMPLE_HZ)
    rng = np.random.default_rng(seed)

    print(f'\n[capture24] Streaming {n_subjects} subjects '
          f'(reservoir cap {max_windows_per_subject:,} windows each) …')
    X, y, segment_df, segment_path = load_all_windows(
        data_dir, n_subjects, max_windows_per_subject, rng,
        rolling_subject=rolling_subject, rolling_rows=rolling_rows)
    print(f'[capture24] {len(X)} windows total')
    for cat in CATEGORIES:
        n = (y == cat).sum()
        if n:
            print(f'            {cat}: {n} windows')

    if len(X) > max_windows_for_stat:
        idx = rng.choice(len(X), max_windows_for_stat, replace=False)
        X_st, y_st = X[idx], y[idx]
    else:
        X_st, y_st = X, y

    print(f'[capture24] Running ADF + KPSS tests on {len(X_st)} windows …')
    adf_frac, kpss_frac = stationarity_summary(X_st, y_st)

    os.makedirs('plots', exist_ok=True)
    out_pdf = 'plots/explore_capture24.pdf'
    print(f'[capture24] Writing {out_pdf} …')

    with PdfPages(out_pdf) as pdf:
        # Page 1: category distribution
        fig, ax = plt.subplots(figsize=(8, 4))
        plot_class_distribution(ax, y, 'CAPTURE24 — category distribution')
        _tight(fig)
        pdf.savefig(fig)
        plt.close(fig)

        # Pages 2+: sample windows per category
        plot_sample_windows(pdf, X, y, 'CAPTURE24')

        # Rolling stats on the most diverse stretch found for one subject
        plot_rolling_stats(pdf, segment_df, segment_path, 'CAPTURE24')

        # Window mean / std distributions
        plot_window_distributions(pdf, X, y, 'CAPTURE24')

        # ACF by category
        plot_acf_by_category(pdf, X, y, 'CAPTURE24')

        # Stationarity heatmaps
        plot_stationarity_heatmaps(pdf, adf_frac, kpss_frac, 'CAPTURE24')

    print(f'[capture24] Done → {out_pdf}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='data/capture24')
    parser.add_argument('--n_subjects', type=int, default=5,
                        help='Number of participants to pool windows from')
    parser.add_argument('--max_windows_per_subject', type=int, default=3000,
                        help='Reservoir-sampling cap per participant '
                             '(unbiased sample across the whole recording)')
    parser.add_argument('--subject', default=None,
                        help='Specific subject CSV (.csv.gz) for the rolling-stats plot '
                             '(default: first subject in the pool)')
    parser.add_argument('--rolling_minutes', type=float, default=30.0,
                        help='Length of the continuous chunk used for rolling stats')
    parser.add_argument('--max_stat', type=int, default=200,
                        help='Max windows for ADF/KPSS (for speed; full pool used for visuals)')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed for reservoir sampling / subsampling')
    args = parser.parse_args()

    run(args.data_dir, args.n_subjects, args.max_windows_per_subject,
        args.subject, args.rolling_minutes, args.max_stat, args.seed)

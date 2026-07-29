"""stationarity_report.py — quick ADF/KPSS stationarity check across all HAR datasets.

Loads X_train directly from preprocessed_data/*.pkl (already windowed to
[N, C, L], consistent across datasets) and runs ADF + KPSS on a random subsample
of windows per channel. Prints a per-dataset summary (mean rejection fraction
across channels) and writes a per-channel detail table to
plots/stationarity_report.tsv.

ADF: low p-value -> reject unit root -> window looks stationary.
KPSS: low p-value -> reject stationarity -> window looks non-stationary.

Usage
-----
  python vis/stationarity_report.py
  python vis/stationarity_report.py --max_stat 500 --seed 1
  python vis/stationarity_report.py --datasets HARTH HAR70plus

Dependencies: numpy, statsmodels
"""

import argparse
import os
import pickle
import warnings

import numpy as np
from statsmodels.tsa.stattools import adfuller, kpss

DATASETS = ['HARTH', 'HAR70plus', 'capture24', 'WISDM', 'WISDM2',
            'USC_HAD', 'Opportunity', 'Skoda']

# Verdict thresholds on the per-dataset mean ADF-stationary / KPSS-non-stationary
# fractions. "Stationary" requires both tests to agree; "Non-stationary" requires
# both tests to agree in the other direction; anything else is a disagreement
# between the two tests and is reported as inconclusive.
STATIONARY_ADF_MIN      = 0.75
STATIONARY_KPSS_MAX      = 0.25
NONSTATIONARY_ADF_MAX    = 0.60
NONSTATIONARY_KPSS_MIN   = 0.30


def classify(adf_frac: float, kpss_frac: float) -> str:
    if adf_frac >= STATIONARY_ADF_MIN and kpss_frac <= STATIONARY_KPSS_MAX:
        return 'stationary'
    if adf_frac <= NONSTATIONARY_ADF_MAX and kpss_frac >= NONSTATIONARY_KPSS_MIN:
        return 'non-stationary'
    return 'inconclusive'


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


def load_train_windows(data_name: str) -> np.ndarray:
    """Load X_train [N, C, L] from preprocessed_data/_DA_{data_name}_256_00.pkl."""
    path = f'preprocessed_data/_DA_{data_name}_256_00.pkl'
    with open(path, 'rb') as f:
        X_train = pickle.load(f)[0]
    return np.asarray(X_train)


def channel_stationarity(X_ch: np.ndarray, alpha: float = 0.05) -> tuple[float, float]:
    """ADF/KPSS rejection fractions for one channel across sampled windows."""
    adf_ps  = [adf_pvalue(X_ch[i])  for i in range(len(X_ch))]
    kpss_ps = [kpss_pvalue(X_ch[i]) for i in range(len(X_ch))]
    adf_frac  = float(np.nanmean(np.array(adf_ps)  < alpha))
    kpss_frac = float(np.nanmean(np.array(kpss_ps) < alpha))
    return adf_frac, kpss_frac


def run(datasets: list, max_stat: int, seed: int, alpha: float):
    rng = np.random.default_rng(seed)
    rows = []  # (dataset, channel, n_windows, adf_frac, kpss_frac)

    for name in datasets:
        path = f'preprocessed_data/_DA_{name}_256_00.pkl'
        if not os.path.isfile(path):
            print(f'[{name}] {path} not found — skipping.')
            continue

        X = load_train_windows(name)   # [N, C, L]
        n_windows, n_ch, _ = X.shape
        n_sample = min(max_stat, n_windows)
        idx = rng.choice(n_windows, n_sample, replace=False)
        X_st = X[idx]

        print(f'[{name}] {n_windows} windows, {n_ch} channels — '
              f'testing {n_sample} sampled windows …')

        for ch in range(n_ch):
            adf_frac, kpss_frac = channel_stationarity(X_st[:, ch, :], alpha)
            rows.append((name, ch, n_sample, adf_frac, kpss_frac))

    if not rows:
        print('No datasets processed.')
        return

    os.makedirs('plots', exist_ok=True)
    out_tsv = 'plots/stationarity_report.tsv'
    with open(out_tsv, 'w') as f:
        f.write('dataset\tchannel\tn_windows\tadf_stationary_frac\tkpss_nonstationary_frac\n')
        for name, ch, n, adf_frac, kpss_frac in rows:
            f.write(f'{name}\t{ch}\t{n}\t{adf_frac:.4f}\t{kpss_frac:.4f}\n')
    print(f'\nPer-channel detail written to {out_tsv}')

    info_lines = [
        'ADF stationary frac:      fraction of windows/channels where ADF rejects a unit root',
        '                          (higher -> more stationary).',
        'KPSS non-stationary frac: fraction where KPSS rejects stationarity',
        '                          (higher -> more non-stationary).',
        '',
        f'Verdict "stationary":     ADF >= {STATIONARY_ADF_MIN} and KPSS <= {STATIONARY_KPSS_MAX}',
        f'Verdict "non-stationary": ADF <= {NONSTATIONARY_ADF_MAX} and KPSS >= {NONSTATIONARY_KPSS_MIN}',
        'Verdict "inconclusive":  the two tests disagree with the above rules',
    ]
    width = max(len(l) for l in info_lines) + 2
    print('\n' + '+' + '-' * width + '+')
    for l in info_lines:
        print('| ' + l.ljust(width - 1) + '|')
    print('+' + '-' * width + '+')

    print(f'\n{"dataset":<14}{"channels":>9}{"adf_stationary":>16}'
          f'{"kpss_nonstationary":>20}{"verdict":>16}')
    for name in datasets:
        ds_rows = [r for r in rows if r[0] == name]
        if not ds_rows:
            continue
        adf_mean  = np.mean([r[3] for r in ds_rows])
        kpss_mean = np.mean([r[4] for r in ds_rows])
        verdict   = classify(adf_mean, kpss_mean)
        print(f'{name:<14}{len(ds_rows):>9}{adf_mean:>16.3f}'
              f'{kpss_mean:>20.3f}{verdict:>16}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets', nargs='+', default=DATASETS,
                        help='Which datasets to test (default: all HAR datasets)')
    parser.add_argument('--max_stat', type=int, default=200,
                        help='Max windows per dataset to run ADF/KPSS on (for speed)')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--alpha', type=float, default=0.05)
    args = parser.parse_args()

    run(args.datasets, args.max_stat, args.seed, args.alpha)

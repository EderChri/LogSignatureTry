"""
data_preprocess.py — Unified preprocessing for all datasets.

Usage:
  python data_preprocess.py                   # process all registered datasets
  python data_preprocess.py --datasets HARTH HAR70plus SleepEEG
"""

import argparse
import glob
import os
import pickle
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------

def get_same_len(data, max_len):
    """Interpolate multidimensional time series (N, D, L) to (N, D, max_len)."""
    N, D, L = data.shape
    data_out = np.zeros((N, D, max_len))
    old_t = np.linspace(0, 1, L)
    new_t = np.linspace(0, 1, max_len)
    for n in range(N):
        for d in range(D):
            data_out[n, d, :] = np.interp(new_t, old_t, data[n, d, :])
    return data_out


# ---------------------------------------------------------------------------
# HARTH / HAR70plus
# ---------------------------------------------------------------------------

HARTH_KEEP_LABELS   = {1, 2, 3, 4, 5, 6, 7, 8, 13, 14, 130, 140}
HAR70PLUS_KEEP_LABELS = {1, 3, 4, 5, 6, 7, 8}
ACCEL_COLS = ['back_x', 'back_y', 'back_z', 'thigh_x', 'thigh_y', 'thigh_z']
WINDOW_LEN = 256
STRIDE = 128
RNG_SEED = 0

# Reference Epilepsy counts for deriving HAR70plus train/val fractions
_EP_TRAIN, _EP_VAL, _EP_TOTAL = 60, 20, 60 + 20 + 11420


def _window_file(fpath: str, keep_labels: set):
    df = pd.read_csv(fpath, usecols=ACCEL_COLS + ['label'])
    values = df[ACCEL_COLS].values.astype(np.float64)
    labels = df['label'].values
    xs, ys = [], []
    for start in range(0, len(values) - WINDOW_LEN + 1, STRIDE):
        end = start + WINDOW_LEN
        unique = set(labels[start:end])
        if len(unique) == 1 and next(iter(unique)) in keep_labels:
            xs.append(values[start:end].T)
            ys.append(int(labels[start]))
    if not xs:
        return (np.empty((0, len(ACCEL_COLS), WINDOW_LEN), dtype=np.float64),
                np.empty((0,), dtype=np.int64))
    return np.stack(xs), np.array(ys, dtype=np.int64)


def _pool_files(file_list: list, keep_labels: set):
    all_x, all_y = [], []
    for fpath in file_list:
        x, y = _window_file(fpath, keep_labels)
        if len(x):
            all_x.append(x)
            all_y.append(y)
    if not all_x:
        return (np.empty((0, len(ACCEL_COLS), WINDOW_LEN), dtype=np.float64),
                np.empty((0,), dtype=np.int64))
    return np.concatenate(all_x), np.concatenate(all_y)


def _remap_labels(arrays: list, keep_labels: set):
    label_map = {orig: idx for idx, orig in enumerate(sorted(keep_labels))}
    remap = np.vectorize(label_map.__getitem__)
    return [remap(y) for y in arrays], label_map


def _stratified_split(X, y, train_frac, val_frac, rng):
    classes = np.unique(y)
    train_idx, val_idx, test_idx = [], [], []
    for cls in classes:
        idx = np.where(y == cls)[0]
        rng.shuffle(idx)
        n = len(idx)
        n_train = max(1, round(n * train_frac))
        n_val   = max(1, round(n * val_frac))
        if n_train + n_val >= n:
            n_train = max(1, n - 1)
            n_val   = min(1, n - n_train)
        train_idx.extend(idx[:n_train])
        val_idx.extend(idx[n_train:n_train + n_val])
        test_idx.extend(idx[n_train + n_val:])
    return (X[train_idx], y[train_idx],
            X[val_idx],   y[val_idx],
            X[test_idx],  y[test_idx])


# ---------------------------------------------------------------------------
# Per-dataset preprocessors
# ---------------------------------------------------------------------------

def preprocess_domain_ts(name: str):
    """Process one of the standard Domain_ts datasets (ECG, EMG, Epilepsy, …)."""
    out = f'preprocessed_data/_DA_{name}_256_00.pkl'
    if os.path.exists(out):
        print(f'Skipping {out}: already exists.')
        return
    src = f'data/Domain_ts/{name}.pkl'
    if not os.path.exists(src):
        print(f'Source not found, skipping: {src}')
        return
    with open(src, 'rb') as f:
        X_train, X_val, X_test, y_train, y_val, y_test = pickle.load(f)
    X_train = get_same_len(X_train, 256)
    X_val   = get_same_len(X_val,   256)
    X_test  = get_same_len(X_test,  256)
    os.makedirs('preprocessed_data', exist_ok=True)
    with open(out, 'wb') as f:
        pickle.dump([X_train, None, None, y_train,
                     X_val,   None, None, y_val,
                     X_test,  None, None, y_test], f)
    print(f'Processed data saved to {out}')


def preprocess_HARTH(data_dir: str = 'harth'):
    out = 'preprocessed_data/_DA_HARTH_256_00.pkl'
    if os.path.exists(out):
        print(f'Skipping {out}: already exists.')
        return
    files = sorted(glob.glob(os.path.join(data_dir, '*.csv')))
    if not files:
        raise FileNotFoundError(f"No CSV files found in '{data_dir}'")
    print(f'HARTH: windowing {len(files)} subjects (all data → X_train, no split)…')
    X_train, y_train = _pool_files(files, HARTH_KEEP_LABELS)
    [y_train], label_map = _remap_labels([y_train], HARTH_KEEP_LABELS)
    D = len(ACCEL_COLS)
    X_empty = np.empty((0, D, WINDOW_LEN), dtype=np.float64)
    y_empty = np.empty((0,), dtype=np.int64)
    print(f'  X_train: {X_train.shape}, classes: {len(label_map)}, map: {label_map}')
    os.makedirs('preprocessed_data', exist_ok=True)
    with open(out, 'wb') as f:
        pickle.dump([X_train, None, None, y_train,
                     X_empty, None, None, y_empty,
                     X_empty, None, None, y_empty], f)
    print(f'  Saved → {out}')


def preprocess_HAR70plus(data_dir: str = 'har70plus'):
    out = 'preprocessed_data/_DA_HAR70plus_256_00.pkl'
    if os.path.exists(out):
        print(f'Skipping {out}: already exists.')
        return
    files = sorted(glob.glob(os.path.join(data_dir, '*.csv')))
    if not files:
        raise FileNotFoundError(f"No CSV files found in '{data_dir}'")
    print(f'HAR70plus: pooling {len(files)} subjects…')
    X_all, y_all = _pool_files(files, HAR70PLUS_KEEP_LABELS)
    total = len(y_all)
    print(f'  Total windows: {total}')
    rng = np.random.default_rng(RNG_SEED)
    X_tr, y_tr, X_va, y_va, X_te, y_te = _stratified_split(
        X_all, y_all, _EP_TRAIN / _EP_TOTAL, _EP_VAL / _EP_TOTAL, rng)
    [y_tr, y_va, y_te], label_map = _remap_labels([y_tr, y_va, y_te], HAR70PLUS_KEEP_LABELS)
    print(f'  train: {len(y_tr)}, val: {len(y_va)}, test: {len(y_te)}  '
          f'({len(y_tr)/total:.2%}/{len(y_va)/total:.2%}/{len(y_te)/total:.2%})')
    print(f'  Train classes: {np.unique(y_tr).tolist()}, Val classes: {np.unique(y_va).tolist()}')
    print(f'  Label map: {label_map}')
    os.makedirs('preprocessed_data', exist_ok=True)
    with open(out, 'wb') as f:
        pickle.dump([X_tr, None, None, y_tr,
                     X_va, None, None, y_va,
                     X_te, None, None, y_te], f)
    print(f'  Saved → {out}')


# ---------------------------------------------------------------------------
# Registry — maps CLI dataset name → callable
# ---------------------------------------------------------------------------

DOMAIN_TS_NAMES = ['ECG', 'EMG', 'Epilepsy', 'FD-B', 'Gesture', 'SleepEEG']

DATASET_REGISTRY = {
    name: (lambda n=name: preprocess_domain_ts(n))
    for name in DOMAIN_TS_NAMES
}
DATASET_REGISTRY['HARTH']    = preprocess_HARTH
DATASET_REGISTRY['HAR70plus'] = preprocess_HAR70plus


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Preprocess datasets')
    parser.add_argument('--datasets', nargs='+', default=list(DATASET_REGISTRY),
                        choices=list(DATASET_REGISTRY),
                        help='Datasets to process (default: all)')
    args = parser.parse_args()

    for name in args.datasets:
        print(f'\n=== {name} ===')
        DATASET_REGISTRY[name]()


if __name__ == '__main__':
    main()

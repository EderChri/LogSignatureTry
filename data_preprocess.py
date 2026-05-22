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
from sklearn.model_selection import train_test_split


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

# Patient-level split sizes for HAR70plus (3 train / 2 val / 13 test out of 18 subjects)
HAR70PLUS_N_TRAIN = 3
HAR70PLUS_N_VAL   = 2


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


def _stratified_patient_split(files: list, keep_labels: set,
                              n_train: int, n_val: int, rng):
    """
    Assign patients to splits so that train and val each cover as many
    activity classes as possible, while keeping each patient's windows
    entirely within one split.

    Strategy (greedy):
    1. Randomly shuffle patients (for reproducibility via rng).
    2. Pick patients for train greedily: prefer those that add new classes
       not yet covered, stop when n_train reached.
    3. Repeat for val from the remaining pool.
    4. Everything else goes to test.
    """
    patient_cls = {}
    for f in files:
        _, y = _window_file(f, keep_labels)
        patient_cls[f] = set(y.tolist())

    pool = list(files)
    rng.shuffle(pool)

    def _greedy_pick(pool, n, covered):
        chosen, remaining = [], []
        # Pass 1: patients that add new classes
        for f in pool:
            if len(chosen) < n and (patient_cls[f] - covered):
                chosen.append(f)
                covered |= patient_cls[f]
            else:
                remaining.append(f)
        # Pass 2: fill remaining slots from random order
        while len(chosen) < n and remaining:
            chosen.append(remaining.pop(0))
        return chosen, remaining, covered

    train_f, pool, cov_tr = _greedy_pick(pool, n_train, set())
    val_f,   pool, _      = _greedy_pick(pool, n_val,   set())
    return train_f, val_f, pool


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
    print(f'HAR70plus: {len(files)} subjects, patient-level split '
          f'({HAR70PLUS_N_TRAIN} train / {HAR70PLUS_N_VAL} val / '
          f'{len(files) - HAR70PLUS_N_TRAIN - HAR70PLUS_N_VAL} test)…')
    rng = np.random.default_rng(RNG_SEED)
    train_files, val_files, test_files = _stratified_patient_split(
        files, HAR70PLUS_KEEP_LABELS, HAR70PLUS_N_TRAIN, HAR70PLUS_N_VAL, rng)
    print(f'  Train subjects: {[os.path.basename(f) for f in train_files]}')
    print(f'  Val subjects:   {[os.path.basename(f) for f in val_files]}')
    print(f'  Test subjects:  {[os.path.basename(f) for f in test_files]}')
    X_tr, y_tr = _pool_files(train_files, HAR70PLUS_KEEP_LABELS)
    X_va, y_va = _pool_files(val_files,   HAR70PLUS_KEEP_LABELS)
    X_te, y_te = _pool_files(test_files,  HAR70PLUS_KEEP_LABELS)
    [y_tr, y_va, y_te], label_map = _remap_labels([y_tr, y_va, y_te], HAR70PLUS_KEEP_LABELS)
    total = len(y_tr) + len(y_va) + len(y_te)
    print(f'  train: {len(y_tr)}, val: {len(y_va)}, test: {len(y_te)}  '
          f'({len(y_tr)/total:.1%}/{len(y_va)/total:.1%}/{len(y_te)/total:.1%})')
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

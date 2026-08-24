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
import torch
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

# Patient-level split sizes for HARTH (31 subjects), matching HAR70plus's
# train/val/test proportions (3/2/13 out of 18 -> ~16.7%/11.1%/72.2%) as closely
# as integer subject counts allow: 5/3/23 out of 31 -> ~16.1%/9.7%/74.2%.
HARTH_N_TRAIN = 5
HARTH_N_VAL   = 3


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


def preprocess_HARTH(data_dir: str = 'data/harth'):
    out = 'preprocessed_data/_DA_HARTH_256_00.pkl'
    if os.path.exists(out):
        print(f'Skipping {out}: already exists.')
        return
    files = sorted(glob.glob(os.path.join(data_dir, '*.csv')))
    if not files:
        raise FileNotFoundError(f"No CSV files found in '{data_dir}'")
    print(f'HARTH: {len(files)} subjects, patient-level split '
          f'({HARTH_N_TRAIN} train / {HARTH_N_VAL} val / '
          f'{len(files) - HARTH_N_TRAIN - HARTH_N_VAL} test)…')
    rng = np.random.default_rng(RNG_SEED)
    train_files, val_files, test_files = _stratified_patient_split(
        files, HARTH_KEEP_LABELS, HARTH_N_TRAIN, HARTH_N_VAL, rng)
    print(f'  Train subjects: {[os.path.basename(f) for f in train_files]}')
    print(f'  Val subjects:   {[os.path.basename(f) for f in val_files]}')
    print(f'  Test subjects:  {[os.path.basename(f) for f in test_files]}')
    X_tr, y_tr = _pool_files(train_files, HARTH_KEEP_LABELS)
    X_va, y_va = _pool_files(val_files,   HARTH_KEEP_LABELS)
    X_te, y_te = _pool_files(test_files,  HARTH_KEEP_LABELS)
    [y_tr, y_va, y_te], label_map = _remap_labels([y_tr, y_va, y_te], HARTH_KEEP_LABELS)
    total = len(y_tr) + len(y_va) + len(y_te)
    print(f'  train: {len(y_tr)}, val: {len(y_va)}, test: {len(y_te)}  '
          f'({len(y_tr)/total:.1%} / {len(y_va)/total:.1%} / {len(y_te)/total:.1%}), '
          f'classes: {len(label_map)}, map: {label_map}')
    os.makedirs('preprocessed_data', exist_ok=True)
    with open(out, 'wb') as f:
        pickle.dump([X_tr, None, None, y_tr,
                     X_va, None, None, y_va,
                     X_te, None, None, y_te], f)
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
# FD-A/B/C/D (bearing fault diagnosis, TF-C benchmark)
# ---------------------------------------------------------------------------
#
# Source: fd_raw/{split}_{domain}.pt — dict with 'samples' [N, 5120] (1-channel
# vibration signal) and 'labels' [N] (3 classes), fixed train/val/test splits,
# native sequence length 5120.
#
# Native length (5120) is far longer than the 256 used by the rest of the
# pipeline, so resampling is unavoidable to reuse the shared architecture and
# tooling — but it's lossy for a vibration signal (~20x downsample discards
# high-frequency content the FFT/derivative views would otherwise use). The
# resampled length is baked into the dataset tag itself (FD-{domain}-256)
# rather than hidden in the "256_00" suffix alone, so it reads as an explicit
# assumption. A future FD-{domain}-ORG variant at native length can be added
# alongside it without ambiguity.

FD_DOMAINS = ['A', 'B', 'C', 'D']
FD_RAW_DIR = 'fd_raw'
FD_SEQ_LEN = 256


def _load_fd_domain_raw(domain: str):
    splits = {}
    for split in ('train', 'val', 'test'):
        fpath = os.path.join(FD_RAW_DIR, f'{split}_{domain.lower()}.pt')
        d = torch.load(fpath, map_location='cpu')
        X = d['samples'].numpy().astype(np.float64)[:, None, :]  # (N, 1, L)
        y = d['labels'].numpy().astype(np.int64)
        splits[split] = (X, y)
    return splits['train'], splits['val'], splits['test']


def preprocess_FD(domain: str, seq_len: int = FD_SEQ_LEN):
    """Bearing fault diagnosis domain {domain} (A/B/C/D), resampled to seq_len."""
    tag = f'FD-{domain}-{seq_len}'
    out = f'preprocessed_data/_DA_{tag}_{seq_len}_00.pkl'
    if os.path.exists(out):
        print(f'Skipping {out}: already exists.')
        return
    if not os.path.isdir(FD_RAW_DIR):
        raise FileNotFoundError(f"Raw FD directory not found: '{FD_RAW_DIR}'")

    (X_tr, y_tr), (X_va, y_va), (X_te, y_te) = _load_fd_domain_raw(domain)

    print(f'FD-{domain}: native {X_tr.shape[2]} -> {seq_len}, '
          f'train {X_tr.shape[0]}, val {X_va.shape[0]}, test {X_te.shape[0]}, '
          f'classes {len(np.unique(y_tr))}')

    X_tr = get_same_len(X_tr, seq_len)
    X_va = get_same_len(X_va, seq_len)
    X_te = get_same_len(X_te, seq_len)

    os.makedirs('preprocessed_data', exist_ok=True)
    with open(out, 'wb') as f:
        pickle.dump([X_tr, None, None, y_tr,
                     X_va, None, None, y_va,
                     X_te, None, None, y_te], f)
    print(f'  Saved → {out}')


# ---------------------------------------------------------------------------
# capture24 (unsupervised pretrain — no labels used)
# ---------------------------------------------------------------------------

CAPTURE24_ACCEL_COLS = ['x', 'y', 'z']


def _window_capture24(values: np.ndarray, seq_len: int, stride: int,
                      rng: np.random.Generator, max_windows: int) -> np.ndarray:
    """Slide a window over a 2-D (N_rows, 3) array, return (K, 3, seq_len)."""
    n_rows = len(values)
    starts = np.arange(0, n_rows - seq_len + 1, stride)
    if len(starts) == 0:
        return np.empty((0, len(CAPTURE24_ACCEL_COLS), seq_len), dtype=np.float32)
    windows = np.stack([values[s:s + seq_len].T for s in starts])  # (K, 3, seq_len)
    # Drop windows that still contain NaN after row-level cleaning
    valid = ~np.isnan(windows).any(axis=(1, 2))
    windows = windows[valid]
    if len(windows) > max_windows:
        idx = rng.choice(len(windows), size=max_windows, replace=False)
        windows = windows[idx]
    return windows.astype(np.float32)


def preprocess_capture24(
    data_dir: str = 'data/capture24',
    seq_len: int = 256,
    stride: int = 128,
    max_per_participant: int = 2000,
    rng_seed: int = 0,
    mini: bool = False,
    mini_n_subjects: int = 10,
    mini_max_per: int = 25,
    mini_rows_limit: int = 1_000_000,
):
    tag = 'capture24mini' if mini else 'capture24'
    out = f'preprocessed_data/_DA_{tag}_256_00.pkl'
    if os.path.exists(out):
        print(f'Skipping {out}: already exists.')
        return

    files = sorted(glob.glob(os.path.join(data_dir, 'P*.csv.gz')))
    if not files:
        raise FileNotFoundError(f"No P*.csv.gz files found in '{data_dir}'")

    if mini:
        files = files[:mini_n_subjects]
        max_per_participant = mini_max_per

    print(f'capture24{"mini" if mini else ""}: processing {len(files)} participants '
          f'(seq_len={seq_len}, stride={stride}, max_per_participant={max_per_participant})…')

    all_windows = []
    rng = np.random.default_rng(rng_seed)

    for i, fpath in enumerate(files):
        pid = os.path.basename(fpath).split('.')[0]
        try:
            # Read only as many rows as needed to yield max_per_participant windows.
            # With stride=128 and seq_len=256, max windows ≈ (nrows - 256) / 128.
            # Adding a 2× safety margin ensures we don't under-sample due to NaN drops.
            nrows_limit = max_per_participant * stride * 2 + seq_len
            kwargs = dict(usecols=CAPTURE24_ACCEL_COLS, low_memory=False, dtype=np.float32,
                          nrows=mini_rows_limit if mini else nrows_limit)
            df = pd.read_csv(fpath, compression='gzip', **kwargs)
        except Exception as e:
            print(f'  Warning: could not read {pid}: {e}')
            continue

        # Drop rows with any NaN in sensor columns before windowing
        df = df.dropna(subset=CAPTURE24_ACCEL_COLS)
        values = df[CAPTURE24_ACCEL_COLS].values  # (N, 3)

        part_rng = np.random.default_rng(rng_seed + i)
        windows = _window_capture24(values, seq_len, stride, part_rng, max_per_participant)

        if len(windows):
            all_windows.append(windows)

        if (i + 1) % 10 == 0 or (i + 1) == len(files):
            total_so_far = sum(len(w) for w in all_windows)
            print(f'  [{i+1}/{len(files)}] {pid}: {len(windows)} windows  '
                  f'(running total: {total_so_far:,})')

    if not all_windows:
        raise RuntimeError('No windows extracted from capture24 data.')

    X_train = np.concatenate(all_windows, axis=0)  # (N, 3, seq_len)
    y_train = np.zeros(len(X_train), dtype=np.int64)

    D = len(CAPTURE24_ACCEL_COLS)
    X_empty = np.empty((0, D, seq_len), dtype=np.float32)
    y_empty = np.empty((0,), dtype=np.int64)

    print(f'  X_train: {X_train.shape}  (dummy y — unsupervised pretrain)')
    os.makedirs('preprocessed_data', exist_ok=True)
    with open(out, 'wb') as f:
        pickle.dump([X_train, None, None, y_train,
                     X_empty, None, None, y_empty,
                     X_empty, None, None, y_empty], f)
    print(f'  Saved → {out}')


# ---------------------------------------------------------------------------
# Generic npy→pkl converter for finetune datasets
# (WISDM, WISDM2, USC_HAD, Opportunity, Skoda)
# ---------------------------------------------------------------------------

_NPY_CONFIGS = {
    # name: (npy_path, n_channels)  — n_channels not used here but documents expected D
    'WISDM':       'data/WISDM/WISDM.npy',
    'WISDM2':      'data/WISDM2/WISDM2.npy',
    'USC_HAD':     'data/USC_HAD/USC_HAD.npy',
    'Opportunity': 'data/Opportunity/Opportunity.npy',
    'Skoda':       'data/Skoda/Skoda.npy',
}

# Fractions: ~17% train / 11% val / 72% test — matches HAR70plus subject ratios
_NPY_TRAIN_FRAC = 0.17
_NPY_VAL_FRAC   = 0.11


def preprocess_npy_dataset(name: str, seq_len: int = 256, rng_seed: int = 0):
    out = f'preprocessed_data/_DA_{name}_256_00.pkl'
    if os.path.exists(out):
        print(f'Skipping {out}: already exists.')
        return

    npy_path = _NPY_CONFIGS.get(name)
    if npy_path is None:
        raise ValueError(f'Unknown npy dataset: {name}')
    if not os.path.exists(npy_path):
        raise FileNotFoundError(f'npy file not found: {npy_path}')

    data = np.load(npy_path, allow_pickle=True).item()
    X_all = np.concatenate([data['train_data'], data['test_data']], axis=0)  # (N, D, L)
    y_all = np.concatenate([data['train_label'], data['test_label']], axis=0)

    # Remap labels to 0-indexed (e.g. USC_HAD is 1-12)
    offset = int(y_all.min())
    if offset != 0:
        y_all = y_all - offset

    # Resample sequence length if needed
    if X_all.shape[2] != seq_len:
        print(f'  Resampling {name} seq_len {X_all.shape[2]} → {seq_len}…')
        X_all = get_same_len(X_all, seq_len)

    # Stratified split: train / val / test
    from sklearn.model_selection import train_test_split
    n_total = len(X_all)
    test_frac = 1.0 - _NPY_TRAIN_FRAC - _NPY_VAL_FRAC  # ~0.72
    idx = np.arange(n_total)

    idx_trainval, idx_test = train_test_split(
        idx, test_size=test_frac, stratify=y_all, random_state=rng_seed)
    # From the trainval portion, split off val
    val_relative = _NPY_VAL_FRAC / (_NPY_TRAIN_FRAC + _NPY_VAL_FRAC)
    idx_train, idx_val = train_test_split(
        idx_trainval, test_size=val_relative, stratify=y_all[idx_trainval],
        random_state=rng_seed)

    X_tr, y_tr = X_all[idx_train], y_all[idx_train]
    X_va, y_va = X_all[idx_val],   y_all[idx_val]
    X_te, y_te = X_all[idx_test],  y_all[idx_test]

    n = len(y_all)
    print(f'{name}: {X_all.shape[1]}ch, seq_len={seq_len}, '
          f'{len(y_tr)} train ({len(y_tr)/n:.1%}) / '
          f'{len(y_va)} val ({len(y_va)/n:.1%}) / '
          f'{len(y_te)} test ({len(y_te)/n:.1%}), '
          f'{len(np.unique(y_tr))} train classes')

    os.makedirs('preprocessed_data', exist_ok=True)
    with open(out, 'wb') as f:
        pickle.dump([X_tr, None, None, y_tr,
                     X_va, None, None, y_va,
                     X_te, None, None, y_te], f)
    print(f'  Saved → {out}')


# ---------------------------------------------------------------------------
# Registry — maps CLI dataset name → callable
# ---------------------------------------------------------------------------

DOMAIN_TS_NAMES = ['ECG', 'EMG', 'Epilepsy', 'Gesture', 'SleepEEG']

DATASET_REGISTRY = {
    name: (lambda n=name: preprocess_domain_ts(n))
    for name in DOMAIN_TS_NAMES
}
DATASET_REGISTRY['HARTH']         = preprocess_HARTH
DATASET_REGISTRY['HAR70plus']     = preprocess_HAR70plus
DATASET_REGISTRY['capture24']     = preprocess_capture24
DATASET_REGISTRY['capture24mini'] = lambda: preprocess_capture24(mini=True)
for _npy_name in _NPY_CONFIGS:
    DATASET_REGISTRY[_npy_name] = lambda n=_npy_name: preprocess_npy_dataset(n)
for _fd_domain in FD_DOMAINS:
    DATASET_REGISTRY[f'FD-{_fd_domain}-{FD_SEQ_LEN}'] = (lambda d=_fd_domain: preprocess_FD(d))


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

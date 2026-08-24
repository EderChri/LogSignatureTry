"""Generate a subject-id sidecar for HARTH or HAR70plus, aligned with the
existing preprocessed_data/_DA_{HARTH,HAR70plus}_256_00.pkl split.

Reuses data_preprocess.py's own windowing/pooling/split functions unchanged
(_window_file, _pool_files, _stratified_patient_split) so the resulting
X/y arrays are bit-identical to what's already in the pkl -- this script only
adds a parallel subject_id array, it does not regenerate or perturb the
existing train/val/test split or any already-trained checkpoint.

Usage:
    python scripts/add_subject_ids.py --dataset harth --raw-dir data/harth
    python scripts/add_subject_ids.py --dataset har70plus --raw-dir data/har70plus

Output:
    preprocessed_data/_DA_{HARTH,HAR70plus}_256_00_subjects.pkl
        {'train': np.ndarray[str], 'val': np.ndarray[str], 'test': np.ndarray[str]}
"""

import argparse
import glob
import importlib.util
import os
import pickle

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
spec = importlib.util.spec_from_file_location(
    'data_preprocess', os.path.join(ROOT, 'scripts', 'data_preprocess.py'))
dp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dp)

DATASETS = {
    'harth': dict(
        tag='HARTH',
        default_raw_dir='data/harth',
        keep_labels=dp.HARTH_KEEP_LABELS,
        n_train=dp.HARTH_N_TRAIN,
        n_val=dp.HARTH_N_VAL,
    ),
    'har70plus': dict(
        tag='HAR70plus',
        default_raw_dir='data/har70plus',
        keep_labels=dp.HAR70PLUS_KEEP_LABELS,
        n_train=dp.HAR70PLUS_N_TRAIN,
        n_val=dp.HAR70PLUS_N_VAL,
    ),
}


def _pool_subject_ids(file_list, keep_labels):
    """Mirror _pool_files, but emit the subject id (file basename) per window
    instead of the windowed signal itself."""
    all_ids = []
    for fpath in file_list:
        x, y = dp._window_file(fpath, keep_labels)
        if len(x):
            subj = os.path.splitext(os.path.basename(fpath))[0]
            all_ids.append(np.full(len(y), subj, dtype=object))
    if not all_ids:
        return np.empty((0,), dtype=object)
    return np.concatenate(all_ids)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True, choices=list(DATASETS))
    ap.add_argument('--raw-dir', default=None,
                    help='Defaults to data/harth or data/har70plus depending on --dataset.')
    args = ap.parse_args()

    cfg = DATASETS[args.dataset]
    raw_dir = args.raw_dir or cfg['default_raw_dir']

    files = sorted(glob.glob(os.path.join(raw_dir, '*.csv')))
    if not files:
        raise FileNotFoundError(f"No CSV files found in '{raw_dir}'")
    print(f'Found {len(files)} subject files in {raw_dir}')

    rng = np.random.default_rng(dp.RNG_SEED)
    train_files, val_files, test_files = dp._stratified_patient_split(
        files, cfg['keep_labels'], cfg['n_train'], cfg['n_val'], rng)
    print(f'  Train subjects: {[os.path.basename(f) for f in train_files]}')
    print(f'  Val subjects:   {[os.path.basename(f) for f in val_files]}')
    print(f'  Test subjects:  {[os.path.basename(f) for f in test_files]}')

    subj_tr = _pool_subject_ids(train_files, cfg['keep_labels'])
    subj_va = _pool_subject_ids(val_files,   cfg['keep_labels'])
    subj_te = _pool_subject_ids(test_files,  cfg['keep_labels'])

    # Sanity check against the existing pkl: same split sizes, same order.
    pkl_path = os.path.join(ROOT, 'preprocessed_data', f'_DA_{cfg["tag"]}_256_00.pkl')
    with open(pkl_path, 'rb') as f:
        existing = pickle.load(f)
    X_tr, _, _, y_tr, X_va, _, _, y_va, X_te, _, _, y_te = existing
    for name, subj, X in [('train', subj_tr, X_tr), ('val', subj_va, X_va), ('test', subj_te, X_te)]:
        if len(subj) != len(X):
            raise RuntimeError(
                f'{name}: subject-id count ({len(subj)}) != existing pkl row count ({len(X)}). '
                'Raw data does not match what produced the existing pkl -- do not proceed blindly.')
    print('Row counts match existing pkl for all splits.')

    out_path = os.path.join(ROOT, 'preprocessed_data', f'_DA_{cfg["tag"]}_256_00_subjects.pkl')
    with open(out_path, 'wb') as f:
        pickle.dump({'train': subj_tr, 'val': subj_va, 'test': subj_te}, f)
    print(f'Saved {out_path}')
    for name, subj in [('train', subj_tr), ('val', subj_va), ('test', subj_te)]:
        u, c = np.unique(subj, return_counts=True)
        print(f'  {name}: {dict(zip(u.tolist(), c.tolist()))}')


if __name__ == '__main__':
    main()

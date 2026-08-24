"""run_finetune.py — unified finetune script for 2-view and 3-view experiments.

Views are determined by --view2 and --view3 (same logic as run_pretrain.py):
  - --view3 omitted (None): 2-view mode
  - --view3 set:            3-view mode

Runs three variants in sequence: finetune, freeze, baseline.
Use --run_modes to select a subset, e.g. --run_modes finetune,freeze.

Usage examples:
  # 3-view
  python scripts/run_finetune.py --data_name _DA_Epilepsy_256_00 \
    --pretrain_data_name _DA_SleepEEG_256_00 \
    --num_feature 1 --num_target 2 --view2 dx --view3 xf \
    --epochs_pretrain 200 --epochs_finetune 100 --seed 0

  # 2-view (xt + logsig)
  python scripts/run_finetune.py --data_name _DA_Epilepsy_256_00 \
    --pretrain_data_name _DA_SleepEEG_256_00 \
    --num_feature 1 --num_target 2 --view2 logsig \
    --encoder_type mlp_logsig --logsig_mode window --logsig_window_size 64 \
    --epochs_pretrain 200 --epochs_finetune 100 --seed 0
"""

import os
import sys
import gc
import pickle

import torch
from torch import nn
from torch.utils.data import DataLoader

from src.config import parse_args
from src.dataloader import (preprocess_data, get_view_num_features,
                             Load_Dataset, _aug_fn_for_view, _make_logsig_noise_aug)
from src.model_nview import EncoderNView, ClassifierNView
from src.trainer import train, test, load_encoder
from src.evaluation import get_clf_metrics
from src.utils import (seed_everything, make_run_tag,
                       write_final_metric_row, log_run_config)

use_cuda = torch.cuda.is_available()
device   = torch.device('cuda' if use_cuda else 'cpu')

args = parse_args()
seed_everything(args.seed)
_run_modes = set(m.strip() for m in args.run_modes.split(','))

if args.pretrain_data_name is None:
    args.pretrain_data_name = args.data_name

args.context_len = int(args.data_name.split('_')[-2])
args.horizon_len = int(args.data_name.split('_')[-1])

# Build view tuple
views = ('xt', args.view2) if args.view3 is None else ('xt', args.view2, args.view3)

# Pretrain checkpoint path (must match what run_pretrain.py produces).
# Does not depend on args.num_feature, so this is safe to compute before the
# feature-dimension bookkeeping below (which needs best_model_path).
pretrain_tag    = make_run_tag(args, views, args.pretrain_data_name)
best_model_path = f'model_pretrain/{args.pretrain_data_name}/{pretrain_tag}.pth'

# ── Load raw data ────────────────────────────────────────────────────────────
print(f'Loading data: preprocessed_data/{args.data_name}.pkl', flush=True)
with open(f'preprocessed_data/{args.data_name}.pkl', 'rb') as f:
    (X_tr_raw, _, _, y_train,
     X_va_raw, _, _, y_val,
     X_te_raw, _, _, y_test) = pickle.load(f)

X_tr = torch.tensor(X_tr_raw).transpose(1, 2).float()
X_va = torch.tensor(X_va_raw).transpose(1, 2).float()
X_te = torch.tensor(X_te_raw).transpose(1, 2).float()
y_tr = torch.tensor(y_train)
y_va = torch.tensor(y_val)
y_te = torch.tensor(y_test)
del X_tr_raw, X_va_raw, X_te_raw
gc.collect()

# Optional subject-id sidecar (currently only HARTH / HAR70plus), aligned
# index-for-index with the test split above — see scripts/add_subject_ids.py.
subject_ids_test = None
_subjects_path = f'preprocessed_data/{args.data_name}_subjects.pkl'
if os.path.exists(_subjects_path):
    with open(_subjects_path, 'rb') as f:
        _subj = pickle.load(f)
    _cand = _subj.get('test')
    if _cand is not None and len(_cand) == len(y_te):
        subject_ids_test = _cand
    else:
        print(f'Warning: {_subjects_path} test-split length mismatch '
              f'({None if _cand is None else len(_cand)} vs {len(y_te)}); '
              f'skipping f1_subject_macro.', flush=True)

# ── Feature-dimension bookkeeping ───────────────────────────────────────────
_pca_k             = getattr(args, 'pca_components', None)
_channel_adapt     = getattr(args, 'channel_adapt', 'none')
_expand_col_assign = None  # set below for channel_adapt=expand
# Always mark the requested strategy in the output filename so that sanity-check runs
# (where channels already match and adaptation is a no-op) produce distinct files from
# channel_adapt=none, allowing the no-op to be verified rather than silently skipped.
_chadapt_suffix = f'_ca{_channel_adapt}' if _channel_adapt != 'none' else ''

if _channel_adapt != 'none' and os.path.exists(best_model_path):
    try:
        _pt_ckpt = torch.load(best_model_path, map_location='cpu', weights_only=False)
    except TypeError:
        _pt_ckpt = torch.load(best_model_path, map_location='cpu')
    _pretrain_num_feature = getattr(_pt_ckpt['args'], 'num_feature', None)
    del _pt_ckpt

    if _pretrain_num_feature is not None and _pretrain_num_feature < args.num_feature:
        if _channel_adapt == 'pca':
            _manual_pca = getattr(args, 'pca_components', None)
            if _manual_pca is not None and _manual_pca != _pretrain_num_feature:
                print(f'[channel_adapt=pca] overriding --pca_components {_manual_pca} -> '
                      f'{_pretrain_num_feature} to match the pretrain checkpoint', flush=True)
            _pca_k          = _pretrain_num_feature
            _chadapt_suffix = f'_capca{_pca_k}'
            print(f'[channel_adapt=pca] {args.num_feature} -> {_pca_k} components '
                  f'(matching pretrain {args.pretrain_data_name})', flush=True)
        elif _channel_adapt == 'drop':
            from src.channel_adapt import select_channels
            _kept_idx = select_channels(args.data_name, args.num_feature,
                                        args.pretrain_data_name, _pretrain_num_feature)
            X_tr, X_va, X_te = X_tr[..., _kept_idx], X_va[..., _kept_idx], X_te[..., _kept_idx]
            args.num_feature = len(_kept_idx)
            _chadapt_suffix  = f'_cadrop{args.num_feature}'
            print(f'[channel_adapt=drop] kept raw channel indices {_kept_idx} '
                  f'(matching pretrain {args.pretrain_data_name})', flush=True)
        elif _channel_adapt == 'expand':
            from src.channel_adapt import assign_pretrain_columns
            _expand_col_assign = assign_pretrain_columns(
                args.data_name, args.num_feature,
                args.pretrain_data_name, _pretrain_num_feature)
            _chadapt_suffix = f'_caexpand{args.num_feature}'
            _unmatched = [fi for fi, pj in enumerate(_expand_col_assign) if pj is None]
            print(f'[channel_adapt=expand] column assignment: {_expand_col_assign} '
                  f'({args.num_feature} finetune channels -> {_pretrain_num_feature} pretrain slots'
                  + (f'; {len(_unmatched)} channels left at random init: {_unmatched}' if _unmatched else '')
                  + ')', flush=True)

if _pca_k is not None and _pca_k < args.num_feature:
    args.num_feature = _pca_k
if args.num_feature > 64:
    args.num_feature = 64

_gt       = getattr(args, 'logsig_global_time', False)
_ft_msp   = getattr(args, 'logsig_multi_smooth_params', None)
_msp_list = [float(p) for p in _ft_msp.split(',')] if _ft_msp else None
_skip_l1  = getattr(args, 'logsig_skip_level1', False)
_ll       = getattr(args, 'logsig_lead_lag', 0)

in_dims = [args.num_feature] + [
    get_view_num_features(v, args.num_feature, args.logsig_depth, _gt, _msp_list, _skip_l1, _ll)
    for v in views[1:]
]
for i, v in enumerate(views[1:], start=2):
    setattr(args, f'num_feature_v{i}', in_dims[i - 1])

# ── Preprocess views ─────────────────────────────────────────────────────────
_logsig_kw = dict(
    logsig_depth=args.logsig_depth,
    logsig_mode=getattr(args, 'logsig_mode', 'stream'),
    logsig_window_size=getattr(args, 'logsig_window_size', 32),
    logsig_smoothing=getattr(args, 'logsig_smoothing', 'tukey'),
    logsig_smooth_param=getattr(args, 'logsig_smooth_param', 0.5),
    logsig_stride=getattr(args, 'logsig_stride', 1),
    logsig_global_time=_gt,
    logsig_multi_smooth_params=_msp_list,
    logsig_normalize=getattr(args, 'logsig_normalize', False),
    logsig_skip_level1=_skip_l1,
    logsig_lead_lag=_ll,
    pca_components=_pca_k,
)

pre_tv = preprocess_data(X_tr, X_va, views=views, **_logsig_kw)
pre_tt = preprocess_data(X_tr, X_te, views=views, **_logsig_kw)

# Use train-fitted normalisation for all splits.
# pre_tv[vi][0] and pre_tt[vi][0] are the same normalised X_train; pick either.
X_train = [pre_tt[f'v{i+1}'][0] for i in range(len(views))]
X_valid = [pre_tv[f'v{i+1}'][1] for i in range(len(views))]
X_test  = [pre_tt[f'v{i+1}'][1] for i in range(len(views))]

# ── Augmentation ─────────────────────────────────────────────────────────────
_logsig_noise_scale = getattr(args, 'logsig_noise_scale', 0.0)
_aug_fns = None
if _logsig_noise_scale > 0.0 and 'logsig' in views:
    _ls_mode  = getattr(args, 'logsig_mode', 'stream')
    _eff_nf   = min((_pca_k if _pca_k is not None else args.num_feature), 64)
    _multi    = _ft_msp is not None and _ls_mode == 'window_smooth'
    _n_copies = len([p.strip() for p in _ft_msp.split(',')]) if _multi else 1
    _logsig_view_idx  = next(i for i, v in enumerate(views) if v == 'logsig')
    _logsig_train_data = X_train[_logsig_view_idx]
    _logsig_aug = _make_logsig_noise_aug(
        _logsig_noise_scale, _logsig_train_data,
        args.logsig_depth, _eff_nf + 1,
        has_global_time=_gt, num_copies=_n_copies,
    )
    _aug_fns = [_logsig_aug if v == 'logsig' else _aug_fn_for_view(v) for v in views]

# ── Dataloaders ───────────────────────────────────────────────────────────────
def _make_loader(X_views, y, mode):
    ds = Load_Dataset(X_views, X_views, y, mode, views=views, aug_fns=_aug_fns)
    return DataLoader(ds, batch_size=args.batch_size_finetune,
                      shuffle=(mode == 'finetune'), drop_last=False,
                      num_workers=4, pin_memory=True, persistent_workers=True)

train_loader = _make_loader(X_train, y_tr, 'finetune')
valid_loader = _make_loader(X_valid, y_va, 'test')
test_loader  = _make_loader(X_test,  y_te, 'test')

os.makedirs(f'out_finetune/{args.data_name}', exist_ok=True)
summary_file      = f'out_finetune/{args.data_name}/final_test_metric_summary.tsv'
monitoring_metric = 'accuracy'

print(args, flush=True)


# ── Single-variant runner ──────────────────────────────────────────────────────
def _run_variant(mode_name: str, load_pretrained: bool):
    output_file = (f'out_finetune/{args.data_name}/'
                   f'{args.data_name}_pt-{pretrain_tag}'
                   f'_{args.feature}_{args.loss_type}_{args.lam}_0{_chadapt_suffix}_{mode_name}')

    if os.path.exists(output_file):
        print(f'Output {output_file} already exists. Skipping.')
        return

    if load_pretrained and not os.path.exists(best_model_path):
        print(f'Pretrained checkpoint not found: {best_model_path}. Skipping {mode_name}.')
        return

    log_run_config(args, phase=f'finetune_{mode_name}', views=views, output_file=output_file)

    encoder = EncoderNView(args, views=list(views), in_dims=in_dims)
    if load_pretrained:
        if _chadapt_suffix and _channel_adapt != 'expand':
            print(f'[channel_adapt={_channel_adapt}] input channels match the pretrain checkpoint '
                  f'({in_dims[0]}) — branches.0.proj weights will transfer, not reinitialise.',
                  flush=True)
        encoder = load_encoder(encoder, best_model_path, new_num_features=None)
        if _channel_adapt == 'expand' and _expand_col_assign is not None:
            # load_encoder skipped branches.0.proj.weight (shape mismatch: pretrain had fewer
            # channels). Patch it now by copying each pretrain column to its assigned finetune
            # column(s). Finetune channels with no valid match (None) keep random init.
            try:
                _ckpt_e = torch.load(best_model_path, map_location='cpu', weights_only=False)
            except TypeError:
                _ckpt_e = torch.load(best_model_path, map_location='cpu')
            _pt_w = _ckpt_e['encoder_state_dict'].get('branches.0.proj.weight')
            if _pt_w is not None:
                _emb_dim, _k = _pt_w.shape
                _new_w = torch.zeros(_emb_dim, in_dims[0], dtype=_pt_w.dtype)
                for _fi, _pj in enumerate(_expand_col_assign):
                    if _pj is not None:
                        _new_w[:, _fi] = _pt_w[:, _pj]
                with torch.no_grad():
                    encoder.branches[0].proj.weight.data.copy_(_new_w)
                print(f'[channel_adapt=expand] patched branches.0.proj.weight '
                      f'{list(_pt_w.shape)} -> {list(_new_w.shape)}', flush=True)
            del _ckpt_e
    encoder = encoder.to(device)
    clf = ClassifierNView(args, views=list(views)).to(device)

    for param in encoder.parameters():
        param.requires_grad = True

    enc_opt = torch.optim.Adam(encoder.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    clf_opt = torch.optim.Adam(clf.parameters(),     lr=args.lr, weight_decay=args.weight_decay)
    enc_sch = torch.optim.lr_scheduler.ReduceLROnPlateau(enc_opt, mode='max', factor=0.5, patience=10)
    clf_sch = torch.optim.lr_scheduler.ReduceLROnPlateau(clf_opt, mode='max', factor=0.5, patience=10)

    # baseline and finetune both train all parameters; only freeze limits encoder
    train_mode = 'freeze' if mode_name == 'freeze' else 'finetune'

    loss_list, metric_list = [], []
    best_valid_mm = 0
    patience, early_stop_counter = 20, 0
    final_test_mm, epochs_trained = None, 0

    for epoch in range(1, args.epochs_finetune + 1):
        train_loss, train_loss_c = train(
            args, encoder, clf, enc_opt, clf_opt, train_loader, mode=train_mode, device=device)
        valid_loss, valid_loss_c = test(
            args, encoder, clf, valid_loader, mode='finetune', device=device)
        test_loss,  test_loss_c  = test(
            args, encoder, clf, test_loader,  mode='finetune', device=device)

        print(f'[{mode_name}] Epoch {epoch}: '
              f'train={train_loss:.4f}/{train_loss_c:.4f}  '
              f'val={valid_loss:.4f}/{valid_loss_c:.4f}  '
              f'test={test_loss:.4f}/{test_loss_c:.4f}', flush=True)
        loss_list.append([train_loss, valid_loss, test_loss,
                          train_loss_c, valid_loss_c, test_loss_c])

        tr_m = get_clf_metrics(args, encoder, clf, train_loader, device)
        va_m = get_clf_metrics(args, encoder, clf, valid_loader, device)
        te_m = get_clf_metrics(args, encoder, clf, test_loader,  device,
                               subject_ids=subject_ids_test)
        _subj_str = f'  f1_subj={te_m["f1_subject_macro"]:.4f}' if te_m['f1_subject_macro'] is not None else ''
        print(f'[{mode_name}] Epoch {epoch}: '
              f'acc train={tr_m["accuracy"]:.4f}  '
              f'val={va_m["accuracy"]:.4f}  '
              f'test={te_m["accuracy"]:.4f}{_subj_str}', flush=True)
        metric_list.append([tr_m, va_m, te_m])
        final_test_mm  = te_m[monitoring_metric]
        epochs_trained = epoch

        enc_sch.step(va_m[monitoring_metric])
        clf_sch.step(va_m[monitoring_metric])

        if va_m[monitoring_metric] > best_valid_mm:
            best_valid_mm      = va_m[monitoring_metric]
            early_stop_counter = 0
        else:
            early_stop_counter += 1

        if early_stop_counter >= patience:
            print(f'[{mode_name}] Early stopping at epoch {epoch}')
            break

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'wb') as f:
        pickle.dump([args, loss_list, metric_list], f)

    if final_test_mm is not None:
        run_name = os.path.basename(output_file)
        write_final_metric_row(summary_file, run_name, final_test_mm, epochs_trained)
        print(f'[{mode_name}] done — score={final_test_mm:.4f}, epochs={epochs_trained}',
              flush=True)


# ── Run variants ──────────────────────────────────────────────────────────────
if 'finetune' in _run_modes:
    _run_variant('finetune',  load_pretrained=True)
else:
    print('Skipping finetune (not in --run_modes)')
if 'freeze'   in _run_modes:
    _run_variant('freeze',   load_pretrained=True)
else:
    print('Skipping freeze (not in --run_modes)')
if 'baseline' in _run_modes:
    _run_variant('baseline', load_pretrained=False)
else:
    print('Skipping baseline (not in --run_modes)')

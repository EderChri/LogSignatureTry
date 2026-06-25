"""run_pretrain.py — unified pretrain script for 2-view and 3-view experiments.

Views are determined by --view2 and --view3:
  - --view3 omitted (None): 2-view mode  → views = ('xt', view2)
  - --view3 set:            3-view mode  → views = ('xt', view2, view3)

Always uses EncoderNView so both cases share the same model code path.
Checkpoint names embed the view combo, so 2-view and 3-view runs never clash.

Usage examples:
  # 3-view
  python scripts/run_pretrain.py --data_name _DA_SleepEEG_256_00 \
    --num_feature 1 --num_target 5 --view2 dx --view3 xf \
    --batch_size_pretrain 64 --epochs_pretrain 200 --seed 0

  # 2-view (xt + logsig)
  python scripts/run_pretrain.py --data_name _DA_SleepEEG_256_00 \
    --num_feature 1 --num_target 5 --view2 logsig \
    --encoder_type mlp_logsig --logsig_mode window --logsig_window_size 64 \
    --batch_size_pretrain 64 --epochs_pretrain 200 --seed 0
"""

import os
import sys
import gc
import pickle
import time

import torch
from torch import nn
from torch.utils.data import DataLoader

from src.config import parse_args
from src.dataloader import (preprocess_data, get_view_num_features,
                             Load_Dataset, _aug_fn_for_view, _make_logsig_noise_aug)
from src.model_nview import EncoderNView
from src.trainer import train, test, load_encoder
from src.utils import (seed_everything, make_run_tag,
                       write_pretrain_summary_row, log_run_config)

use_cuda = torch.cuda.is_available()
device   = torch.device('cuda' if use_cuda else 'cpu')

args = parse_args()
seed_everything(args.seed)
run_start_time = time.time()

# Build view tuple — view3=None means 2-view mode
views = ('xt', args.view2) if args.view3 is None else ('xt', args.view2, args.view3)

print(
    f"Starting pretrain: data={args.data_name}, encoder={args.encoder_type}, "
    f"views={views}, epochs={args.epochs_pretrain}, "
    f"batch={args.batch_size_pretrain}, seed={args.seed}",
    flush=True,
)

args.context_len = int(args.data_name.split('_')[-2])
args.horizon_len = int(args.data_name.split('_')[-1])

# Determine the tag used in paths — may differ from args.data_name for full_training
_data_tag = f'{args.data_name}-full' if getattr(args, 'full_training', False) else args.data_name
run_tag      = make_run_tag(args, views, _data_tag)
output_file  = f'out_pretrain/{_data_tag}/{run_tag}'
resume_ckpt_path = f'out_pretrain/.resume_{run_tag}.pth'

if os.path.exists(output_file):
    print(f'Output {output_file} already exists. Skipping.')
    sys.exit(0)

# Log config before any expensive work so interrupted runs appear in history
log_run_config(args, phase='pretrain', views=views, output_file=output_file)

# ── Load raw data ────────────────────────────────────────────────────────────
print(f'Loading data: preprocessed_data/{args.data_name}.pkl', flush=True)
with open(f'preprocessed_data/{args.data_name}.pkl', 'rb') as f:
    (X_train_intp, X_train_shrink, X_train_forecast,
     y_train,
     X_val_intp,   X_val_shrink,   X_val_forecast,
     y_val,
     X_test_intp,  X_test_shrink,  X_test_forecast,
     y_test) = pickle.load(f)

import numpy as np, random as _random
import torch

X_train_intp = torch.as_tensor(X_train_intp).transpose(1, 2).float()
y_train      = torch.as_tensor(y_train)
del X_train_shrink, X_train_forecast
del X_val_shrink,  X_val_forecast
del X_test_shrink, X_test_forecast

if getattr(args, 'full_training', False):
    X_val_intp  = torch.as_tensor(X_val_intp).transpose(1, 2).float()
    y_val       = torch.as_tensor(y_val)
    X_test_intp = torch.as_tensor(X_test_intp).transpose(1, 2).float()
    y_test      = torch.as_tensor(y_test)
    X_train_intp = torch.cat([X_train_intp, X_val_intp, X_test_intp], dim=0)
    y_train      = torch.cat([y_train, y_val, y_test], dim=0)

del X_val_intp, y_val, X_test_intp, y_test
gc.collect()

# ── Feature-dimension bookkeeping ────────────────────────────────────────────
_pca_k    = getattr(args, 'pca_components', None)
if _pca_k is not None and _pca_k < args.num_feature:
    args.num_feature = _pca_k
if args.num_feature > 64:
    args.num_feature = 64

_gt      = getattr(args, 'logsig_global_time', False)
_ls_msp  = getattr(args, 'logsig_multi_smooth_params', None)
_msp_list = [float(p) for p in _ls_msp.split(',')] if _ls_msp else None

in_dims = [args.num_feature] + [
    get_view_num_features(v, args.num_feature, args.logsig_depth, _gt, _msp_list)
    for v in views[1:]
]
# Keep per-view attrs on args for downstream use / checkpoints
for i, v in enumerate(views[1:], start=2):
    setattr(args, f'num_feature_v{i}', in_dims[i - 1])

# ── Preprocess views ─────────────────────────────────────────────────────────
_ls_mode   = getattr(args, 'logsig_mode', 'stream')
_ls_wsiz   = getattr(args, 'logsig_window_size', 32)
_ls_smooth = getattr(args, 'logsig_smoothing', 'tukey')
_ls_sp     = getattr(args, 'logsig_smooth_param', 0.5)
_ls_stride = getattr(args, 'logsig_stride', 1)
_ls_norm   = getattr(args, 'logsig_normalize', False)
_msp_key   = ('_msp' + _ls_msp.replace(',', '-')) if _ls_msp else ''
_pca_key   = f'_pca{_pca_k}' if _pca_k else ''
_norm_key  = '_norm' if _ls_norm else ''
_logsig_cache_key = (
    f'{_data_tag}_d{args.logsig_depth}_{_ls_mode}'
    f'_w{_ls_wsiz}_s{_ls_stride}_{_ls_smooth}_sp{_ls_sp}_gt{int(_gt)}{_msp_key}{_pca_key}{_norm_key}'
)

print(f'Preprocessing views: {views}', flush=True)
preprocess_start = time.time()
preprocessed = preprocess_data(
    X_train_intp, X_train_intp, views=views,
    logsig_depth=args.logsig_depth,
    logsig_mode=_ls_mode, logsig_window_size=_ls_wsiz,
    logsig_smoothing=_ls_smooth, logsig_smooth_param=_ls_sp,
    logsig_stride=_ls_stride, logsig_global_time=_gt,
    logsig_multi_smooth_params=_msp_list,
    logsig_normalize=_ls_norm,
    logsig_cache_key=_logsig_cache_key,
    pca_components=_pca_k,
)
X_train     = [preprocessed[f'v{i+1}'][0] for i in range(len(views))]
X_train_aug = list(X_train)
print(f'Preprocessing done in {(time.time()-preprocess_start)/60:.2f} min', flush=True)

# Optional time-lag augmentation for logsig view
_ls_lag = getattr(args, 'logsig_lag', 0)
if _ls_lag > 0 and 'logsig' in views:
    _logsig_view_idx = next(i for i, v in enumerate(views) if v == 'logsig')
    _logsig_data     = X_train[_logsig_view_idx]
    _lagged          = torch.zeros_like(_logsig_data)
    _lagged[:, _ls_lag:, :] = _logsig_data[:, :-_ls_lag, :]
    X_train_aug[_logsig_view_idx] = _lagged
    print(f'[logsig_lag={_ls_lag}] Time-lagged logsig as positive for view {_logsig_view_idx}', flush=True)

# Optional per-level logsig noise augmentation
_logsig_noise_scale = getattr(args, 'logsig_noise_scale', 0.0)
_aug_fns = None
if _logsig_noise_scale > 0.0 and 'logsig' in views:
    _eff_nf  = min((_pca_k if _pca_k is not None else args.num_feature), 64)
    _multi   = _ls_msp is not None and _ls_mode == 'window_smooth'
    _n_copies = len([p.strip() for p in _ls_msp.split(',')]) if _multi else 1
    _logsig_view_idx = next(i for i, v in enumerate(views) if v == 'logsig')
    _logsig_aug = _make_logsig_noise_aug(
        _logsig_noise_scale, X_train[_logsig_view_idx],
        args.logsig_depth, _eff_nf + 1,
        has_global_time=_gt, num_copies=_n_copies,
    )
    _aug_fns = [_logsig_aug if v == 'logsig' else _aug_fn_for_view(v) for v in views]

# ── Datasets and loaders ─────────────────────────────────────────────────────
print('Building datasets and dataloaders', flush=True)
pretrain_dataset = Load_Dataset(X_train, X_train_aug, y_train, 'pretrain',
                                views=views, aug_fns=_aug_fns)
prevalid_dataset = Load_Dataset(X_train, X_train_aug, y_train, 'pretrain',
                                views=views, aug_fns=_aug_fns)
pretrain_loader = DataLoader(pretrain_dataset, batch_size=args.batch_size_pretrain,
                             shuffle=True,  drop_last=False,
                             num_workers=4, pin_memory=True, persistent_workers=True)
prevalid_loader = DataLoader(prevalid_dataset, batch_size=args.batch_size_pretrain,
                             shuffle=False, drop_last=False,
                             num_workers=4, pin_memory=True, persistent_workers=True)

# ── Model ─────────────────────────────────────────────────────────────────────
os.makedirs(f'model_pretrain/{_data_tag}', exist_ok=True)
os.makedirs(f'out_pretrain/{_data_tag}',   exist_ok=True)
summary_file    = f'out_pretrain/{_data_tag}/final_pretrain_summary.tsv'
best_model_path = f'model_pretrain/{_data_tag}/{run_tag}.pth'

if torch.cuda.device_count() > 1:
    encoder = EncoderNView(args, views=list(views), in_dims=in_dims)
    encoder = nn.DataParallel(encoder).to(device)
else:
    encoder = EncoderNView(args, views=list(views), in_dims=in_dims).to(device)
    if hasattr(torch, 'compile'):
        _cc = torch.cuda.get_device_capability(device)
        if _cc[0] >= 8:
            try:
                encoder = torch.compile(encoder)
                print('torch.compile applied to encoder', flush=True)
            except Exception as e:
                print(f'torch.compile skipped: {e}', flush=True)
        else:
            print(f'torch.compile skipped: compute {_cc[0]}.{_cc[1]} < 8.0', flush=True)

if getattr(args, 'random_attn_init', False):
    for m in encoder.modules():
        if isinstance(m, torch.nn.MultiheadAttention):
            torch.nn.init.normal_(m.in_proj_weight, std=1.0)
            if m.out_proj.weight is not None:
                torch.nn.init.normal_(m.out_proj.weight, std=1.0)
    print('random_attn_init: MHA weights reinitialised with N(0,1)', flush=True)

encoder_optimizer = torch.optim.Adam(encoder.parameters(), lr=args.lr,
                                     weight_decay=args.weight_decay)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    encoder_optimizer, mode='min', factor=0.5, patience=10)

# ── Resume interrupted run ────────────────────────────────────────────────────
loss_list         = []
best_valid_loss   = float('inf')
patience          = 20
early_stop_counter = 0
epoch_start       = 1

if os.path.exists(resume_ckpt_path):
    print(f'Loading resume checkpoint: {resume_ckpt_path}', flush=True)
    ckpt = torch.load(resume_ckpt_path, map_location='cpu', weights_only=False)
    epoch_start        = ckpt['epoch'] + 1
    best_valid_loss    = ckpt['best_valid_loss']
    loss_list          = ckpt['loss_list']
    early_stop_counter = ckpt['early_stop_counter']
    encoder.load_state_dict(ckpt['encoder_state_dict'], strict=False)
    encoder_optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    scheduler.load_state_dict(ckpt['scheduler_state_dict'])
    print(f'Resumed from epoch {epoch_start}, best_valid_loss={best_valid_loss:.4f}', flush=True)

# ── Training loop ─────────────────────────────────────────────────────────────
print(args, flush=True)
epoch_durations = []
epochs_trained  = epoch_start - 1
_enc_raw = encoder.module if isinstance(encoder, (nn.DataParallel, nn.parallel.DistributedDataParallel)) else encoder

for epoch in range(epoch_start, args.epochs_pretrain + 1):
    epoch_t0   = time.time()
    train_loss = train(args, encoder, None, encoder_optimizer, None,
                       pretrain_loader, mode='pretrain', device=device)
    with torch.no_grad():
        valid_loss = test(args, encoder, None, prevalid_loader,
                          mode='pretrain', device=device)
    scheduler.step(valid_loss)

    elapsed = time.time() - epoch_t0
    epoch_durations.append(elapsed)
    eta = (sum(epoch_durations) / len(epoch_durations)) * (args.epochs_pretrain - epoch) / 60
    print(
        f'Epoch {epoch}: train={train_loss:.4f}  val={valid_loss:.4f}  '
        f'({elapsed:.1f}s  ETA {eta:.1f}min)',
        flush=True,
    )
    epochs_trained = epoch
    loss_list.append([train_loss, valid_loss])

    if valid_loss < best_valid_loss:
        best_valid_loss    = valid_loss
        early_stop_counter = 0
        print(f'[Saving best model at epoch {epoch}, val_loss={valid_loss:.4f}]', flush=True)
        torch.save({
            'epoch':               epoch,
            'args':                args,
            'views':               list(views),
            'encoder_state_dict':  _enc_raw.state_dict(),
            'optimizer_state_dict': encoder_optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'loss_list':           loss_list,
            'best_valid_loss':     best_valid_loss,
        }, best_model_path)
    else:
        early_stop_counter += 1

    # Named epoch checkpoint (--save_every N)
    save_every = getattr(args, 'save_every', 0)
    if save_every > 0 and epoch % save_every == 0:
        eckpt_dir = f'model_pretrain/{_data_tag}/epoch_ckpts'
        os.makedirs(eckpt_dir, exist_ok=True)
        torch.save({'epoch': epoch, 'args': args, 'views': list(views),
                    'encoder_state_dict': _enc_raw.state_dict()},
                   f'{eckpt_dir}/{run_tag}_ep{epoch}.pth')

    # Resume checkpoint saved every epoch for safe interruption
    os.makedirs(os.path.dirname(resume_ckpt_path), exist_ok=True)
    torch.save({
        'epoch':               epoch,
        'encoder_state_dict':  _enc_raw.state_dict(),
        'optimizer_state_dict': encoder_optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'loss_list':           loss_list,
        'best_valid_loss':     best_valid_loss,
        'early_stop_counter':  early_stop_counter,
    }, resume_ckpt_path)

    if early_stop_counter >= patience:
        print(f'Early stopping at epoch {epoch}')
        break

# ── Save results ──────────────────────────────────────────────────────────────
os.makedirs(os.path.dirname(output_file), exist_ok=True)
with open(output_file, 'wb') as f:
    pickle.dump([args, loss_list], f)

if os.path.exists(resume_ckpt_path):
    os.remove(resume_ckpt_path)

run_name = os.path.basename(output_file)
write_pretrain_summary_row(summary_file, run_name, best_valid_loss, epochs_trained)
print(f'Done. best_valid_loss={best_valid_loss:.4f}, epochs={epochs_trained}', flush=True)
print(f'Total runtime: {(time.time()-run_start_time)/60:.2f} min', flush=True)

import os
import sys
import math
import random
import argparse
import pickle 
import time
import gc
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

import torch
from torch import nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import torchcde
from pytorch_metric_learning import losses

from src.config import *
from src.dataloader import *
from src.model import *
from src.trainer import *
from src.evaluation import *
from src.utils import *

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")


def write_pretrain_summary_row(summary_path, run_name, best_valid_loss, epochs_trained):
    if not os.path.exists(summary_path):
        with open(summary_path, 'w') as f:
            f.write('run_name\tbest_valid_loss\tepochs_trained\n')

    with open(summary_path, 'a') as f:
        f.write(f'{run_name}\t{best_valid_loss:.6f}\t{epochs_trained}\n')


def seed_everything(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    
args = parse_args()
seed_everything(args.seed)
run_start_time = time.time()

_enc_suffix = f'_{args.encoder_type}' if args.encoder_type != 'transformer' else ''

def _logsig_suffix(args) -> str:
    """Suffix encoding logsig mode + window size — empty for default stream mode."""
    mode = getattr(args, 'logsig_mode', 'stream')
    if mode == 'stream':
        base = ''
    else:
        wsiz = getattr(args, 'logsig_window_size', 32)
        if mode == 'window':
            base = f'_win{wsiz}'
        else:
            smoothing    = getattr(args, 'logsig_smoothing', 'tukey')
            smooth_param = getattr(args, 'logsig_smooth_param', 0.5)
            msp          = getattr(args, 'logsig_multi_smooth_params', None)
            base = f'_{smoothing}{wsiz}'   # e.g. _tukey32  _ema16
            if msp:
                k = len([p.strip() for p in msp.split(',')])
                base += f'_msp{k}'
            elif smooth_param != 0.5:
                base += f'_sp{smooth_param}'
    stride = getattr(args, 'logsig_stride', 1)
    gt     = getattr(args, 'logsig_global_time', False)
    pool   = getattr(args, 'logsig_pool', 'auto')
    depth  = getattr(args, 'logsig_depth', 2)
    if stride > 1:
        base += f'_s{stride}'
    if gt:
        base += '_gt'
    if pool != 'auto':
        base += f'_p{pool}'   # e.g. _plast  _pmean
    if depth != 2:
        base += f'_d{depth}'
    lsn = getattr(args, 'logsig_noise_scale', 0.0)
    if lsn > 0.0:
        base += f'_lsn{lsn}'
    if getattr(args, 'logsig_normalize', False):
        base += '_norm'
    lag = getattr(args, 'logsig_lag', 0)
    if lag > 0:
        base += f'_lag{lag}'
    return base

_lsig_suffix = _logsig_suffix(args)
_il_suffix = '' if getattr(args, 'interaction_type', 'attention') == 'attention' \
             else f'_il{args.interaction_type.replace("_", "")}'
_cv_suffix = f'_cv{getattr(args, "lam_cross", 1.0)}' if getattr(args, 'cross_view_logsig', False) else ''

print(
    f"Starting pretrain: data={args.data_name}, encoder={args.encoder_type}, "
    f"view2={args.view2}, view3={args.view3}, "
    f"epochs={args.epochs_pretrain}, batch={args.batch_size_pretrain}, seed={args.seed}",
    flush=True,
)

## Check if output already exists
_data_tag = f'{args.data_name}-full' if args.full_training else args.data_name
output_file = f'out_pretrain/{args.data_name}/{_data_tag}_v2{args.view2}_v3{args.view3}_ep{args.epochs_pretrain}_{args.seed}{_enc_suffix}{_lsig_suffix}{_il_suffix}{_cv_suffix}'

if os.path.exists(output_file):
    print(f"Output file {output_file} already exists. Skipping this run.")
    sys.exit(0)

# Resume checkpoint path for interrupted runs
resume_ckpt_path = f'out_pretrain/.resume_{args.data_name}_v2{args.view2}_v3{args.view3}_ep{args.epochs_pretrain}_{args.seed}{_enc_suffix}{_lsig_suffix}{_il_suffix}{_cv_suffix}.pth'
    
#
args.context_len = int(args.data_name.split('_')[-2])
args.horizon_len = int(args.data_name.split('_')[-1])

## Load data
print(f"Loading data: preprocessed_data/{args.data_name}.pkl", flush=True)
with open(f'preprocessed_data/{args.data_name}.pkl', 'rb') as f:
    X_train_intp, X_train_shirink, X_train_forecast, y_train, X_val_intp, X_val_shirink, X_val_forecast, y_val, X_test_intp, X_test_shirink, X_test_forecast, y_test = pickle.load(f)

## Keep only tensors actually needed for pretraining.
X_train_intp = torch.as_tensor(X_train_intp).transpose(1, 2).float()
y_train = torch.as_tensor(y_train)

# Drop unused raw arrays early to lower peak host memory.
del X_train_shirink, X_train_forecast
del X_val_shirink, X_val_forecast
del X_test_shirink, X_test_forecast

##
if args.full_training:
    X_val_intp = torch.as_tensor(X_val_intp).transpose(1, 2).float()
    y_val = torch.as_tensor(y_val)
    X_test_intp = torch.as_tensor(X_test_intp).transpose(1, 2).float()
    y_test = torch.as_tensor(y_test)

    X_train_intp = torch.cat([X_train_intp, X_val_intp, X_test_intp], dim=0)
    y_train = torch.cat([y_train, y_val, y_test], dim=0)
    args.data_name = '-'.join([args.data_name, 'full'])

# Free val/test arrays when not used further.
del X_val_intp, y_val, X_test_intp, y_test
gc.collect()

##
views = ('xt', args.view2, args.view3)

print(f"Preprocessing views: {views} (logsig_depth={args.logsig_depth})", flush=True)
preprocess_start_time = time.time()

_ls_mode   = getattr(args, 'logsig_mode', 'stream')
_ls_wsiz   = getattr(args, 'logsig_window_size', 32)
_ls_smooth = getattr(args, 'logsig_smoothing', 'tukey')
_ls_sp     = getattr(args, 'logsig_smooth_param', 0.5)
_ls_stride = getattr(args, 'logsig_stride', 1)
_ls_gt     = getattr(args, 'logsig_global_time', False)
_ls_norm   = getattr(args, 'logsig_normalize', False)
_ls_msp    = getattr(args, 'logsig_multi_smooth_params', None)
_msp_key   = ('_msp' + _ls_msp.replace(',', '-')) if _ls_msp else ''
_pca_k     = getattr(args, 'pca_components', None)
_pca_key   = f'_pca{_pca_k}' if _pca_k else ''
_norm_key  = '_norm' if _ls_norm else ''
_logsig_cache_key = (
    f'{args.data_name}_d{args.logsig_depth}_{_ls_mode}'
    f'_w{_ls_wsiz}_s{_ls_stride}_{_ls_smooth}_sp{_ls_sp}_gt{int(_ls_gt)}{_msp_key}{_pca_key}{_norm_key}'
)
preprocessed_data = preprocess_data(
    X_train_intp, X_train_intp, views=views,
    logsig_depth=args.logsig_depth,
    logsig_mode=_ls_mode, logsig_window_size=_ls_wsiz,
    logsig_smoothing=_ls_smooth, logsig_smooth_param=_ls_sp,
    logsig_stride=_ls_stride, logsig_global_time=_ls_gt,
    logsig_multi_smooth_params=[float(p) for p in _ls_msp.split(',')] if _ls_msp else None,
    logsig_normalize=_ls_norm,
    logsig_cache_key=_logsig_cache_key,
    pca_components=_pca_k,
)
X_train_intp_v1, _, _, _ = preprocessed_data['v1']
X_train_intp_v2, _, _, _ = preprocessed_data['v2']
X_train_intp_v3, _, _, _ = preprocessed_data['v3']

X_train = [X_train_intp_v1, X_train_intp_v2, X_train_intp_v3]
# X_train_aug is intentionally identical to X_train here; augmentations are
# applied in the dataset pipeline. Reusing the same tensors avoids a second,
# memory-heavy preprocess pass.
X_train_aug = list(X_train)

_ls_lag = getattr(args, 'logsig_lag', 0)
if _ls_lag > 0 and 'logsig' in views:
    _logsig_view_idx = next(i for i, v in enumerate(views) if v == 'logsig')
    _logsig_data = X_train[_logsig_view_idx]          # [N, L, C]
    _logsig_lag = torch.zeros_like(_logsig_data)
    _logsig_lag[:, _ls_lag:, :] = _logsig_data[:, :-_ls_lag, :]
    X_train_aug[_logsig_view_idx] = _logsig_lag
    print(f'[logsig_lag={_ls_lag}] Using time-lagged logsig as positive augmentation for view {_logsig_view_idx + 1}', flush=True)
preprocess_elapsed = time.time() - preprocess_start_time
print(f"Preprocessing finished in {preprocess_elapsed / 60:.2f} min", flush=True)

##
_logsig_noise_scale = getattr(args, 'logsig_noise_scale', 0.0)
_aug_fns = None
if _logsig_noise_scale > 0.0 and 'logsig' in views and not getattr(args, 'cross_view_logsig', False):
    _eff_nf = min((_pca_k if _pca_k is not None else args.num_feature), 64)
    _multi_aug = _ls_msp is not None and _ls_mode == 'window_smooth'
    _num_copies = len([p.strip() for p in _ls_msp.split(',')]) if _multi_aug else 1
    _logsig_view_idx = next(i for i, v in enumerate(views) if v == 'logsig')
    _logsig_aug = _make_logsig_noise_aug(
        _logsig_noise_scale, X_train[_logsig_view_idx],
        args.logsig_depth, _eff_nf + 1,
        has_global_time=_ls_gt, num_copies=_num_copies,
    )
    _aug_fns = [_logsig_aug if v == 'logsig' else _aug_fn_for_view(v) for v in views]

print("Building datasets and dataloaders", flush=True)
pretrain_dataset = Load_Dataset(X_train, X_train_aug, y_train, 'pretrain', views=views, aug_fns=_aug_fns)
prevalid_dataset = Load_Dataset(X_train, X_train_aug, y_train, 'prevalid', views=views, aug_fns=_aug_fns)
pretrain_loader = DataLoader(pretrain_dataset, batch_size=args.batch_size_pretrain, shuffle=True, drop_last=False, num_workers=4, pin_memory=True, persistent_workers=True)
prevalid_loader = DataLoader(prevalid_dataset, batch_size=args.batch_size_pretrain, shuffle=False, drop_last=False, num_workers=4, pin_memory=True, persistent_workers=True)

##
os.makedirs(f'model_pretrain', exist_ok=True)
os.makedirs(f'model_pretrain/{args.data_name}', exist_ok=True)

os.makedirs(f'out_pretrain', exist_ok=True)
os.makedirs(f'out_pretrain/{args.data_name}', exist_ok=True)
summary_file = f'out_pretrain/{args.data_name}/final_pretrain_summary.tsv'

# Apply PCA dimension override before computing model architecture sizes
if _pca_k is not None and _pca_k < args.num_feature:
    args.num_feature = _pca_k
if args.num_feature > 64:
    args.num_feature = 64

_gt = getattr(args, 'logsig_global_time', False)
_msp_list = [float(p) for p in _ls_msp.split(',')] if _ls_msp else None
args.num_feature_v2 = get_view_num_features(args.view2, args.num_feature, args.logsig_depth, _gt, _msp_list)
args.num_feature_v3 = get_view_num_features(args.view3, args.num_feature, args.logsig_depth, _gt, _msp_list)

if torch.cuda.device_count() > 1:
    encoder = Encoder(args)
    encoder = nn.DataParallel(encoder).to(device)
else:
    encoder = Encoder(args).to(device)
    if hasattr(torch, 'compile'):
        _cc = torch.cuda.get_device_capability(device)
        if _cc[0] >= 8:
            try:
                encoder = torch.compile(encoder)
                print('torch.compile applied to encoder', flush=True)
            except Exception as e:
                print(f'torch.compile skipped: {e}', flush=True)
        else:
            print(f'torch.compile skipped: compute {_cc[0]}.{_cc[1]} < 8.0 (Triton SDPA instability)', flush=True)
if getattr(args, 'random_attn_init', False):
    for m in encoder.modules():
        if isinstance(m, torch.nn.MultiheadAttention):
            torch.nn.init.normal_(m.in_proj_weight, std=1.0)
            if m.out_proj.weight is not None:
                torch.nn.init.normal_(m.out_proj.weight, std=1.0)
    print('random_attn_init: MHA projection weights reinitialised with N(0,1)', flush=True)
encoder_optimizer = torch.optim.Adam(encoder.parameters(), lr=args.lr, weight_decay=args.weight_decay)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(encoder_optimizer, mode='min', factor=0.5, patience=10)

loss_list = []
best_valid_loss = float('inf')
best_model_path = f'model_pretrain/{args.data_name}/{args.data_name}_v2{args.view2}_v3{args.view3}_ep{args.epochs_pretrain}_{args.seed}{_enc_suffix}{_lsig_suffix}{_il_suffix}{_cv_suffix}.pth'
output_file = f'out_pretrain/{args.data_name}/{args.data_name}_v2{args.view2}_v3{args.view3}_ep{args.epochs_pretrain}_{args.seed}{_enc_suffix}{_lsig_suffix}{_il_suffix}{_cv_suffix}'

# Early stopping parameters
patience = 20
early_stop_counter = 0

# Check for resume checkpoint
epoch_start = 1
if os.path.exists(resume_ckpt_path):
    print(f"Loading resume checkpoint: {resume_ckpt_path}", flush=True)
    resume_state = torch.load(resume_ckpt_path, map_location='cpu', weights_only=False)
    
    # Validate num_feature matches
    resume_num_feature = resume_state.get('num_feature')
    if resume_num_feature is not None and resume_num_feature != args.num_feature:
        print(
            f"Error: Resume checkpoint expects num_feature={resume_num_feature} "
            f"but args.num_feature={args.num_feature}. Please re-run with matching num_feature.",
            file=sys.stderr,
        )
        sys.exit(1)
    
    epoch_start = resume_state['epoch'] + 1
    best_valid_loss = resume_state['best_valid_loss']
    loss_list = resume_state['loss_list']
    early_stop_counter = resume_state['early_stop_counter']
    encoder.load_state_dict(resume_state['encoder_state_dict'], strict=False)
    encoder_optimizer.load_state_dict(resume_state['optimizer_state_dict'])
    scheduler.load_state_dict(resume_state['scheduler_state_dict'])
    print(f"Resumed from epoch {epoch_start}, best_valid_loss={best_valid_loss:.4f}", flush=True)

print(args)
epoch_durations = []
epochs_trained = epoch_start - 1
for epoch in range(epoch_start, args.epochs_pretrain + 1):
    epoch_start_time = time.time()
    encoder.train()
    train_loss = train(args, encoder, None, encoder_optimizer, None, pretrain_loader, mode='pretrain', device=device)
    
    encoder.eval()
    with torch.no_grad():
        valid_loss = test(args, encoder, None, prevalid_loader, mode='pretrain', device=device)
    
    scheduler.step(valid_loss)

    epoch_elapsed = time.time() - epoch_start_time
    epoch_durations.append(epoch_elapsed)
    avg_epoch_sec = sum(epoch_durations) / len(epoch_durations)
    remaining_epochs = args.epochs_pretrain - epoch
    eta_minutes = (avg_epoch_sec * remaining_epochs) / 60
    
    print(
        f'Epoch {epoch}: Train Loss = {train_loss:.4f}, Validation Loss = {valid_loss:.4f}, '
        f'Epoch Time = {epoch_elapsed:.1f}s, ETA = {eta_minutes:.2f} min',
        flush=True,
    )
    epochs_trained = epoch
    loss_list.append([train_loss, valid_loss])
    
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        early_stop_counter = 0
        print(f'[Saving model at epoch {epoch} with validation loss {valid_loss:.4f}]')
        
        # Save the model state, optimizer state, and current epoch
        torch.save({
            'epoch': epoch,
            'args': args,
            # 'encoder_state_dict': encoder.state_dict(),
            'encoder_state_dict': encoder.module.state_dict() if isinstance(encoder, (nn.DataParallel, nn.parallel.DistributedDataParallel)) else encoder.state_dict(),
            'optimizer_state_dict': encoder_optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'loss_list': loss_list,
            'best_valid_loss': best_valid_loss
        }, best_model_path)
    else:
        early_stop_counter += 1
    
    # Save named epoch checkpoint (for attention inspection / learning-curve analysis)
    save_every = getattr(args, 'save_every', 0)
    if save_every > 0 and epoch % save_every == 0:
        eckpt_dir = f'model_pretrain/{args.data_name}/epoch_ckpts'
        os.makedirs(eckpt_dir, exist_ok=True)
        run_tag = os.path.basename(best_model_path).replace('.pth', '')
        eckpt = f'{eckpt_dir}/{run_tag}_ep{epoch}.pth'
        state = encoder.module.state_dict() if isinstance(encoder, (nn.DataParallel, nn.parallel.DistributedDataParallel)) else encoder.state_dict()
        torch.save({'epoch': epoch, 'encoder_state_dict': state, 'args': args}, eckpt)

    # Save resume checkpoint after each epoch for safe interruption
    os.makedirs(os.path.dirname(resume_ckpt_path), exist_ok=True)
    torch.save({
        'epoch': epoch,
        'encoder_state_dict': encoder.module.state_dict() if isinstance(encoder, (nn.DataParallel, nn.parallel.DistributedDataParallel)) else encoder.state_dict(),
        'optimizer_state_dict': encoder_optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'loss_list': loss_list,
        'best_valid_loss': best_valid_loss,
        'early_stop_counter': early_stop_counter,
    }, resume_ckpt_path)
    
    # Check for early stopping
    if early_stop_counter >= patience:
        print(f'Early stopping triggered at epoch {epoch}')
        break

# Save final results
os.makedirs(os.path.dirname(output_file), exist_ok=True)
with open(output_file, 'wb') as f:
    pickle.dump([args, loss_list], f)

# Clean up resume checkpoint after successful completion
if os.path.exists(resume_ckpt_path):
    os.remove(resume_ckpt_path)
    print(f"Cleaned up resume checkpoint", flush=True)

run_name = os.path.basename(output_file)
write_pretrain_summary_row(summary_file, run_name, best_valid_loss, epochs_trained)
print(f"Pretrain summary written: {run_name}, best_valid_loss={best_valid_loss:.4f}, epochs={epochs_trained}", flush=True)

total_elapsed = time.time() - run_start_time
print(f"Training completed. Best validation loss: {best_valid_loss:.4f}")
print(f"Total runtime: {total_elapsed / 60:.2f} min", flush=True)

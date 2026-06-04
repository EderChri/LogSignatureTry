"""run_finetune_nview.py — 2-view finetune: xt + view2.

Companion to run_pretrain_nview.py.  Loads a pretrained EncoderNView checkpoint
and runs three variants: finetune, freeze, baseline.

Usage (same flags as run_finetune.py; --view3 is ignored):
  python run_finetune_nview.py \
      --data_name _DA_Epilepsy_256_00 \
      --pretrain_data_name _DA_SleepEEG_256_00 \
      --num_feature 1 --num_target 2 \
      --view2 logsig \
      --encoder_type mlp_logsig \
      --logsig_mode window --logsig_window_size 64 \
      --epochs_pretrain 200 --epochs_finetune 100 --seed 0
"""

import os
import sys
import gc
import random
import pickle

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.config import parse_args
from src.dataloader import preprocess_data, get_view_num_features
from src.dataloader_nview import Load_DatasetNView
from src.model_nview import EncoderNView, ClassifierNView
from src.trainer_nview import train_nview, test_nview
from src.trainer import load_encoder
from src.evaluation import get_clf_metrics_nview
from src.utils import *

use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')


def write_final_metric_row(summary_path, run_name, final_test_score, epochs_trained):
    if not os.path.exists(summary_path):
        with open(summary_path, 'w') as f:
            f.write('run_name\tfinal_test_score\tepochs_trained\n')
    with open(summary_path, 'a') as f:
        f.write(f'{run_name}\t{final_test_score:.6f}\t{epochs_trained}\n')


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
_run_modes = set(m.strip() for m in args.run_modes.split(','))

_enc_suffix = f'_{args.encoder_type}' if args.encoder_type != 'transformer' else ''


def _logsig_suffix(args) -> str:
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
            base = f'_{smoothing}{wsiz}'
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
        base += f'_p{pool}'
    if depth != 2:
        base += f'_d{depth}'
    return base


_lsig_suffix = _logsig_suffix(args)
_il_suffix   = '' if getattr(args, 'interaction_type', 'attention') == 'attention' \
               else f'_il{args.interaction_type.replace("_", "")}'

if args.pretrain_data_name is None:
    args.pretrain_data_name = args.data_name

views = ('xt', args.view2)

args.context_len = int(args.data_name.split('_')[3])
args.horizon_len = int(args.data_name.split('_')[4])

# Feature dims
_pca_k = getattr(args, 'pca_components', None)
if _pca_k is not None and _pca_k < args.num_feature:
    args.num_feature = _pca_k
if args.num_feature > 64:
    args.num_feature = 64
_gt     = getattr(args, 'logsig_global_time', False)
_ft_msp = getattr(args, 'logsig_multi_smooth_params', None)
_ft_msp_list = [float(p) for p in _ft_msp.split(',')] if _ft_msp else None
args.num_feature_v2 = get_view_num_features(args.view2, args.num_feature, args.logsig_depth, _gt, _ft_msp_list)
args.loss_type = 'ALL'
args.num_views = 2

# Pretrain checkpoint
pretrain_tag = (f'{args.pretrain_data_name}_v2{args.view2}_nview'
                f'_ep{args.epochs_pretrain}_{args.seed}{_enc_suffix}{_lsig_suffix}{_il_suffix}')
best_model_path = f'model_pretrain/{args.pretrain_data_name}/{pretrain_tag}.pth'

# Load data
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

_ft_msp = getattr(args, 'logsig_multi_smooth_params', None)
_logsig_kw = dict(
    logsig_depth=args.logsig_depth,
    logsig_mode=getattr(args, 'logsig_mode', 'stream'),
    logsig_window_size=getattr(args, 'logsig_window_size', 32),
    logsig_smoothing=getattr(args, 'logsig_smoothing', 'tukey'),
    logsig_smooth_param=getattr(args, 'logsig_smooth_param', 0.5),
    logsig_stride=getattr(args, 'logsig_stride', 1),
    logsig_global_time=_gt,
    logsig_multi_smooth_params=[float(p) for p in _ft_msp.split(',')] if _ft_msp else None,
    pca_components=_pca_k,
)

pre_tv = preprocess_data(X_tr, X_va, views=views, **_logsig_kw)
pre_tt = preprocess_data(X_tr, X_te, views=views, **_logsig_kw)

Xtr1, Xva1 = pre_tv['v1'][0], pre_tv['v1'][1]
Xtr2, Xva2 = pre_tv['v2'][0], pre_tv['v2'][1]
Xtr1, Xte1 = pre_tt['v1'][0], pre_tt['v1'][1]
Xtr2, Xte2 = pre_tt['v2'][0], pre_tt['v2'][1]

def make_loader(v1, v2, y, mode):
    ds = Load_DatasetNView([v1, v2], [v1, v2], y, mode, views=list(views))
    return DataLoader(ds, batch_size=args.batch_size_finetune, shuffle=(mode == 'finetune'),
                      drop_last=False, num_workers=4, pin_memory=True, persistent_workers=True)

train_loader = make_loader(Xtr1, Xtr2, y_tr, 'finetune')
valid_loader = make_loader(Xva1, Xva2, y_va, 'test')
test_loader  = make_loader(Xte1, Xte2, y_te, 'test')

os.makedirs(f'out_finetune/{args.data_name}', exist_ok=True)
summary_file     = f'out_finetune/{args.data_name}/final_test_metric_summary.tsv'
monitoring_metric = 'accuracy'
in_dims           = [args.num_feature, args.num_feature_v2]

print(args, flush=True)


def _run_variant(mode_name, load_pretrained: bool):
    """Run one finetune/freeze/baseline variant."""
    output_file = (f'out_finetune/{args.data_name}/'
                   f'{args.data_name}_pt-{pretrain_tag}'
                   f'_{args.feature}_{args.loss_type}_{args.lam}_0_{mode_name}')

    if os.path.exists(output_file):
        print(f'Output {output_file} already exists. Skipping.')
        return

    if load_pretrained and not os.path.exists(best_model_path):
        print(f'Pretrained checkpoint not found: {best_model_path}. Skipping {mode_name}.')
        return

    encoder = EncoderNView(args, views=list(views), in_dims=in_dims)
    if load_pretrained:
        encoder = load_encoder(encoder, best_model_path, new_num_features=None)
    encoder = encoder.to(device)
    clf = ClassifierNView(args, views=list(views)).to(device)

    for param in encoder.parameters():
        param.requires_grad = True

    enc_opt = torch.optim.Adam(encoder.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    clf_opt = torch.optim.Adam(clf.parameters(),     lr=args.lr, weight_decay=args.weight_decay)
    enc_sch = torch.optim.lr_scheduler.ReduceLROnPlateau(enc_opt, mode='max', factor=0.5, patience=10)
    clf_sch = torch.optim.lr_scheduler.ReduceLROnPlateau(clf_opt, mode='max', factor=0.5, patience=10)

    # mode passed to train_nview: baseline uses 'finetune' (all params enabled, random init)
    train_mode = 'freeze' if mode_name == 'freeze' else 'finetune'

    loss_list, metric_list = [], []
    best_valid_mm = 0
    patience, early_stop_counter = 20, 0
    final_test_mm, epochs_trained = None, 0

    for epoch in range(1, args.epochs_finetune + 1):
        train_loss, train_loss_c = train_nview(
            args, encoder, clf, enc_opt, clf_opt, train_loader, mode=train_mode, device=device)
        valid_loss, valid_loss_c = test_nview(
            args, encoder, clf, valid_loader, mode='finetune', device=device)
        test_loss,  test_loss_c  = test_nview(
            args, encoder, clf, test_loader,  mode='finetune', device=device)

        print(f'[{mode_name}] Epoch {epoch}: '
              f'train={train_loss:.4f}/{train_loss_c:.4f}  '
              f'val={valid_loss:.4f}/{valid_loss_c:.4f}  '
              f'test={test_loss:.4f}/{test_loss_c:.4f}', flush=True)
        loss_list.append([train_loss, valid_loss, test_loss,
                          train_loss_c, valid_loss_c, test_loss_c])

        tr_m  = get_clf_metrics_nview(args, encoder, clf, train_loader, device)
        va_m  = get_clf_metrics_nview(args, encoder, clf, valid_loader, device)
        te_m  = get_clf_metrics_nview(args, encoder, clf, test_loader,  device)
        print(f'[{mode_name}] Epoch {epoch}: '
              f'acc train={tr_m["accuracy"]:.4f}  '
              f'val={va_m["accuracy"]:.4f}  '
              f'test={te_m["accuracy"]:.4f}', flush=True)
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
            print(f'Early stopping at epoch {epoch}')
            break

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'wb') as f:
        pickle.dump([args, loss_list, metric_list], f)

    if final_test_mm is not None:
        run_name = os.path.basename(output_file)
        write_final_metric_row(summary_file, run_name, final_test_mm, epochs_trained)
        print(f'[{mode_name}] done — score={final_test_mm:.4f}, epochs={epochs_trained}', flush=True)


_run_variant('finetune',  load_pretrained=True)
if 'freeze'   in _run_modes: _run_variant('freeze',   load_pretrained=True)
else: print("Skipping freeze (not in --run_modes)")
if 'baseline' in _run_modes: _run_variant('baseline', load_pretrained=False)
else: print("Skipping baseline (not in --run_modes)")

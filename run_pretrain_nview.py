"""run_pretrain_nview.py — 2-view pretrain: xt + view2.

Drop-in companion to run_pretrain.py for experiments with exactly two views
(raw time series + one other).  Uses EncoderNView / Load_DatasetNView from the
nview modules.  Checkpoint tags include '_nview' so they never overwrite
3-view checkpoints.

Usage (same flags as run_pretrain.py; --view3 is ignored):
  python run_pretrain_nview.py \
      --data_name _DA_HARTH_256_00 \
      --num_feature 6 --num_target 4 \
      --view2 logsig \
      --encoder_type mlp_logsig \
      --logsig_mode window --logsig_window_size 64 \
      --batch_size_pretrain 64 --epochs_pretrain 200 --seed 0
"""

import os
import sys
import gc
import math
import random
import pickle
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.config import parse_args
from src.dataloader import preprocess_data, get_view_num_features
from src.dataloader_nview import Load_DatasetNView
from src.model_nview import EncoderNView
from src.trainer_nview import train_nview, test_nview
from src.trainer import load_encoder
from src.utils import *

use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')


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
    mode = getattr(args, 'logsig_mode', 'stream')
    if mode == 'stream':
        base = ''
    else:
        wsiz = getattr(args, 'logsig_window_size', 32)
        if mode == 'window':
            base = f'_win{wsiz}'
        else:
            smoothing = getattr(args, 'logsig_smoothing', 'tukey')
            base = f'_{smoothing}{wsiz}'
    stride = getattr(args, 'logsig_stride', 1)
    gt     = getattr(args, 'logsig_global_time', False)
    if stride > 1:
        base += f'_s{stride}'
    if gt:
        base += '_gt'
    return base


_lsig_suffix = _logsig_suffix(args)

# 2-view: xt + view2 only
views = ('xt', args.view2)

print(
    f"Starting 2-view pretrain: data={args.data_name}, encoder={args.encoder_type}, "
    f"views={views}, epochs={args.epochs_pretrain}, batch={args.batch_size_pretrain}, "
    f"seed={args.seed}",
    flush=True,
)

args.context_len = int(args.data_name.split('_')[3])
args.horizon_len = int(args.data_name.split('_')[4])

_data_tag = f'{args.data_name}-full' if getattr(args, 'full_training', False) else args.data_name
output_file = (f'out_pretrain/{args.data_name}/'
               f'{_data_tag}_v2{args.view2}_nview'
               f'_ep{args.epochs_pretrain}_{args.seed}{_enc_suffix}{_lsig_suffix}')

if os.path.exists(output_file):
    print(f'Output {output_file} already exists. Skipping.')
    sys.exit(0)

resume_ckpt_path = (f'out_pretrain/.resume_{args.data_name}_v2{args.view2}_nview'
                    f'_ep{args.epochs_pretrain}_{args.seed}{_enc_suffix}{_lsig_suffix}.pth')

# Load data
print(f'Loading data: preprocessed_data/{args.data_name}.pkl', flush=True)
with open(f'preprocessed_data/{args.data_name}.pkl', 'rb') as f:
    (X_train_intp, X_train_shirink, X_train_forecast,
     y_train, X_val_intp, X_val_shirink, X_val_forecast,
     y_val, X_test_intp, X_test_shirink, X_test_forecast, y_test) = pickle.load(f)

X_train_intp = torch.as_tensor(X_train_intp).transpose(1, 2).float()
y_train      = torch.as_tensor(y_train)
del X_train_shirink, X_train_forecast
del X_val_intp, X_val_shirink, X_val_forecast
del X_test_intp, X_test_shirink, X_test_forecast, y_val, y_test
gc.collect()

# Preprocess views
_gt = getattr(args, 'logsig_global_time', False)
if args.num_feature > 64:
    args.num_feature = 64

args.num_feature_v2 = get_view_num_features(args.view2, args.num_feature, args.logsig_depth, _gt)
# ClassifierNView uses args.loss_type for fc input size; always ALL for nview
args.loss_type = 'ALL'
args.num_views = 2

print(f'Preprocessing views: {views}', flush=True)
preprocess_start = time.time()
preprocessed = preprocess_data(
    X_train_intp, X_train_intp, views=views,
    logsig_depth=args.logsig_depth,
    logsig_mode=getattr(args, 'logsig_mode', 'stream'),
    logsig_window_size=getattr(args, 'logsig_window_size', 32),
    logsig_smoothing=getattr(args, 'logsig_smoothing', 'tukey'),
    logsig_smooth_param=getattr(args, 'logsig_smooth_param', 0.5),
    logsig_stride=getattr(args, 'logsig_stride', 1),
    logsig_global_time=_gt,
)
v1_tr = preprocessed['v1'][0]
v2_tr = preprocessed['v2'][0]
print(f'Preprocessing done in {(time.time()-preprocess_start)/60:.2f} min', flush=True)

X_train     = [v1_tr, v2_tr]
X_train_aug = X_train

# Datasets and loaders
pretrain_dataset = Load_DatasetNView(X_train, X_train_aug, y_train, 'pretrain', views=list(views))
prevalid_dataset = Load_DatasetNView(X_train, X_train_aug, y_train, 'pretrain', views=list(views))
pretrain_loader = DataLoader(pretrain_dataset, batch_size=args.batch_size_pretrain,
                             shuffle=True,  drop_last=False,
                             num_workers=4, pin_memory=True, persistent_workers=True)
prevalid_loader = DataLoader(prevalid_dataset, batch_size=args.batch_size_pretrain,
                             shuffle=False, drop_last=False,
                             num_workers=4, pin_memory=True, persistent_workers=True)

# Model
os.makedirs(f'model_pretrain/{args.data_name}', exist_ok=True)
os.makedirs(f'out_pretrain/{args.data_name}', exist_ok=True)
summary_file  = f'out_pretrain/{args.data_name}/final_pretrain_summary.tsv'
best_model_path = (f'model_pretrain/{args.data_name}/'
                   f'{args.data_name}_v2{args.view2}_nview'
                   f'_ep{args.epochs_pretrain}_{args.seed}{_enc_suffix}{_lsig_suffix}.pth')

in_dims = [args.num_feature, args.num_feature_v2]
encoder = EncoderNView(args, views=list(views), in_dims=in_dims).to(device)
encoder_optimizer = torch.optim.Adam(encoder.parameters(), lr=args.lr,
                                     weight_decay=args.weight_decay)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    encoder_optimizer, mode='min', factor=0.5, patience=10)

loss_list        = []
best_valid_loss  = float('inf')
patience         = 20
early_stop_counter = 0
epoch_start      = 1

# Resume interrupted run
if os.path.exists(resume_ckpt_path):
    print(f'Resuming from {resume_ckpt_path}', flush=True)
    ckpt = torch.load(resume_ckpt_path, map_location='cpu', weights_only=False)
    encoder.load_state_dict(ckpt['encoder_state_dict'])
    encoder_optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    epoch_start      = ckpt['epoch'] + 1
    best_valid_loss  = ckpt.get('best_valid_loss', float('inf'))
    early_stop_counter = ckpt.get('early_stop_counter', 0)
    loss_list        = ckpt.get('loss_list', [])
    print(f'Resumed at epoch {epoch_start}', flush=True)

print('Training', flush=True)
epochs_trained = 0
for epoch in range(epoch_start, args.epochs_pretrain + 1):
    train_loss = train_nview(args, encoder, None, encoder_optimizer, None,
                             pretrain_loader, mode='pretrain', device=device)
    valid_loss = test_nview(args,  encoder, None,
                             prevalid_loader, mode='pretrain', device=device)
    scheduler.step(valid_loss)
    loss_list.append([train_loss, valid_loss])
    epochs_trained = epoch
    print(f'Epoch {epoch}: train={train_loss:.4f}  valid={valid_loss:.4f}', flush=True)

    if valid_loss < best_valid_loss:
        best_valid_loss    = valid_loss
        early_stop_counter = 0
        torch.save({'encoder_state_dict': encoder.state_dict()}, best_model_path)
    else:
        early_stop_counter += 1

    if early_stop_counter >= patience:
        print(f'Early stopping at epoch {epoch}')
        break

    # Save resume checkpoint every 10 epochs
    if epoch % 10 == 0:
        torch.save({
            'epoch': epoch,
            'encoder_state_dict': encoder.state_dict(),
            'optimizer_state_dict': encoder_optimizer.state_dict(),
            'best_valid_loss': best_valid_loss,
            'early_stop_counter': early_stop_counter,
            'loss_list': loss_list,
        }, resume_ckpt_path)

# Save output marker and clean up resume checkpoint
with open(output_file, 'wb') as f:
    pickle.dump([args, loss_list], f)

if os.path.exists(resume_ckpt_path):
    os.remove(resume_ckpt_path)

run_name = os.path.basename(output_file)
write_pretrain_summary_row(summary_file, run_name, best_valid_loss, epochs_trained)
print(f'Done. best_valid_loss={best_valid_loss:.4f}, epochs={epochs_trained}', flush=True)

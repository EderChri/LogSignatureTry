"""View masking diagnostic: quantifies each view's contribution to classification.

Loads a pretrained encoder, trains a lightweight classifier head in freeze mode
(only classifier + encoder input projections update), then evaluates test accuracy
while zeroing out each view's embedding in turn.

The accuracy drop when masking view i is a direct measure of how much that view
contributes to the model's predictions.

Usage (same args as run_finetune.py):
  python scripts/view_masking.py \
    --data_name _DA_Epilepsy_256_00 \
    --pretrain_data_name _DA_SleepEEG_256_00 \
    --num_feature 1 --num_target 2 \
    --view2 logsig --encoder_type mlp_logsig --interaction_type bilinear \
    --logsig_mode window --logsig_window_size 64 \
    --epochs_pretrain 200 --epochs_finetune 50 --seed 0

  # Point directly at an existing checkpoint (bypasses auto-generated path):
  python scripts/view_masking.py ... --checkpoint path/to/encoder.pth
"""

import gc
import os
import pickle
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse as _ap
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Strip --checkpoint before parse_args so it doesn't trip the unknown-arg check.
_pre = _ap.ArgumentParser(add_help=False)
_pre.add_argument('--checkpoint', default=None)
_known, _remaining = _pre.parse_known_args()
sys.argv = [sys.argv[0]] + _remaining

from src.config import parse_args
from src.dataloader import (preprocess_data, get_view_num_features,
                             Load_Dataset, _aug_fn_for_view, _make_logsig_noise_aug)
from src.model_nview import EncoderNView, ClassifierNView
from src.model import _pool
from src.utils import seed_everything, make_run_tag

use_cuda = torch.cuda.is_available()
device   = torch.device('cuda' if use_cuda else 'cpu')

args = parse_args()
seed_everything(args.seed)

if args.pretrain_data_name is None:
    args.pretrain_data_name = args.data_name

args.context_len = int(args.data_name.split('_')[-2])
args.horizon_len = int(args.data_name.split('_')[-1])

views = ('xt', args.view2) if args.view3 is None else ('xt', args.view2, args.view3)

_pca_k = getattr(args, 'pca_components', None)
if _pca_k is not None and _pca_k < args.num_feature:
    args.num_feature = _pca_k
if args.num_feature > 64:
    args.num_feature = 64

_gt      = getattr(args, 'logsig_global_time', False)
_ft_msp  = getattr(args, 'logsig_multi_smooth_params', None)
_msp_list = [float(p) for p in _ft_msp.split(',')] if _ft_msp else None
_skip_l1  = getattr(args, 'logsig_skip_level1', False)
_ll       = getattr(args, 'logsig_lead_lag', 0)

in_dims = [args.num_feature] + [
    get_view_num_features(v, args.num_feature, args.logsig_depth, _gt, _msp_list, _skip_l1, _ll)
    for v in views[1:]
]
for i, v in enumerate(views[1:], start=2):
    setattr(args, f'num_feature_v{i}', in_dims[i - 1])

if _known.checkpoint:
    best_model_path = _known.checkpoint
else:
    pretrain_tag    = make_run_tag(args, views, args.pretrain_data_name)
    best_model_path = f'model_pretrain/{args.pretrain_data_name}/{pretrain_tag}.pth'


# ── Inlined load_encoder (avoids importing src/trainer → pytorch_metric_learning → scipy) ──
def _load_encoder(encoder, checkpoint_path):
    try:
        state_dict = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    except TypeError:
        state_dict = torch.load(checkpoint_path, map_location='cpu')

    if isinstance(state_dict, dict) and 'encoder_state_dict' in state_dict:
        state_dict = state_dict['encoder_state_dict']

    # Strip DataParallel / torch.compile prefixes
    cleaned = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            k = k[len('module.'):]
        elif k.startswith('_orig_mod.'):
            k = k[len('_orig_mod.'):]
        cleaned[k] = v
    state_dict = cleaned

    # Remap old Encoder (model.py) key names → EncoderNView (model_nview.py) key names.
    # Checkpoints saved before the EncoderNView refactor use named branches
    # (input_layer_t / transformer_encoder_t / output_layer_t for view 0,
    #  input_layer_d / transformer_encoder_d / output_layer_d for view 1,
    #  input_layer_f / transformer_encoder_f / output_layer_f for view 2).
    # EncoderNView stores them as branches.{i}.{proj|pe|enc} / output_layers.{i}.
    _old_to_new = {}
    view_map = {'t': '0', 'd': '1', 'f': '2'}
    for letter, idx in view_map.items():
        # input projection: input_layer_X → branches.X.proj  (transformer branch)
        _old_to_new[f'input_layer_{letter}.weight'] = f'branches.{idx}.proj.weight'
        _old_to_new[f'input_layer_{letter}.bias']   = f'branches.{idx}.proj.bias'
        # input MLP: input_layer_X.net.* → branches.X.net.*  (MLP branch)
        for suffix in state_dict:
            if suffix.startswith(f'input_layer_{letter}.net.'):
                rest = suffix[len(f'input_layer_{letter}.'):]
                _old_to_new[suffix] = f'branches.{idx}.{rest}'
        # transformer encoder body
        for suffix in state_dict:
            if suffix.startswith(f'transformer_encoder_{letter}.'):
                rest = suffix[len(f'transformer_encoder_{letter}.'):]
                _old_to_new[suffix] = f'branches.{idx}.enc.{rest}'
        # output layers: output_layer_X.* → output_layers.X.*
        for suffix in state_dict:
            if suffix.startswith(f'output_layer_{letter}.'):
                rest = suffix[len(f'output_layer_{letter}.'):]
                _old_to_new[suffix] = f'output_layers.{idx}.{rest}'
    # positional encoding: shared in old code, per-branch in new
    for idx in view_map.values():
        _old_to_new['positional_encoding.pe'] = f'branches.{idx}.pe.pe'

    remapped = {}
    for k, v in state_dict.items():
        new_k = _old_to_new.get(k, k)
        # Only keep the first mapping for positional_encoding.pe
        if new_k not in remapped:
            remapped[new_k] = v
    state_dict = remapped

    # Sanitise NaN/Inf
    for k, v in state_dict.items():
        if torch.is_tensor(v) and (torch.isnan(v).any() or torch.isinf(v).any()):
            state_dict[k] = torch.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)

    state_dict = {k: v for k, v in state_dict.items() if 'q_func' not in k}

    model_sd = encoder.state_dict()
    matched   = {k: v for k, v in state_dict.items()
                 if k in model_sd and model_sd[k].shape == v.shape}
    skipped   = [k for k, v in state_dict.items()
                 if k in model_sd and model_sd[k].shape != v.shape]

    print(f'  Checkpoint keys matched : {len(matched)} / {len(model_sd)} model keys')
    if skipped:
        print(f'  Re-initialized (shape mismatch): {skipped}')

    encoder.load_state_dict(matched, strict=False)
    return encoder


def _is_input_layer(name: str) -> bool:
    parts = name.split('.')
    return len(parts) >= 3 and parts[0] == 'branches' and parts[2] in ('proj', 'net')


# ── Load raw data ─────────────────────────────────────────────────────────────
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

# ── Preprocess views ──────────────────────────────────────────────────────────
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

X_train = [pre_tt[f'v{i+1}'][0] for i in range(len(views))]
X_valid = [pre_tv[f'v{i+1}'][1] for i in range(len(views))]
X_test  = [pre_tt[f'v{i+1}'][1] for i in range(len(views))]

# ── Augmentation ──────────────────────────────────────────────────────────────
_logsig_noise_scale = getattr(args, 'logsig_noise_scale', 0.0)
_aug_fns = None
if _logsig_noise_scale > 0.0 and 'logsig' in views:
    _ls_mode  = getattr(args, 'logsig_mode', 'stream')
    _eff_nf   = min((_pca_k if _pca_k is not None else args.num_feature), 64)
    _multi    = _ft_msp is not None and _ls_mode == 'window_smooth'
    _n_copies = len([p.strip() for p in _ft_msp.split(',')]) if _multi else 1
    _logsig_view_idx   = next(i for i, v in enumerate(views) if v == 'logsig')
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

# ── Build and load encoder ────────────────────────────────────────────────────
print(f'\nLoading pretrained encoder: {best_model_path}', flush=True)
encoder = EncoderNView(args, views=list(views), in_dims=in_dims)
encoder = _load_encoder(encoder, best_model_path)
encoder = encoder.to(device)

clf = ClassifierNView(args, views=list(views)).to(device)

# ── Train classifier (freeze mode, cross-entropy only) ────────────────────────
enc_opt = torch.optim.Adam(encoder.parameters(), lr=args.lr, weight_decay=args.weight_decay)
clf_opt = torch.optim.Adam(clf.parameters(),     lr=args.lr, weight_decay=args.weight_decay)
enc_sch = torch.optim.lr_scheduler.ReduceLROnPlateau(enc_opt, mode='max', factor=0.5, patience=10)
clf_sch = torch.optim.lr_scheduler.ReduceLROnPlateau(clf_opt, mode='max', factor=0.5, patience=10)

cls_crit = nn.CrossEntropyLoss()

best_val_acc   = 0.0
best_enc_sd    = None
best_clf_sd    = None
patience_cnt   = 0
patience_limit = 20

print(f'\nTraining classifier (freeze mode) for up to {args.epochs_finetune} epochs ...\n',
      flush=True)

for epoch in range(1, args.epochs_finetune + 1):
    encoder.eval()
    for name, param in encoder.named_parameters():
        param.requires_grad = _is_input_layer(name)
    clf.train()

    for batch in train_loader:
        batch      = [t.float().to(device) for t in batch]
        views_orig = batch[:len(views)]
        y          = batch[2 * len(views)].long()

        enc_opt.zero_grad()
        clf_opt.zero_grad()

        hiddens, projs = encoder(*views_orig)
        inputs = projs if args.feature == 'latent' else hiddens
        loss   = cls_crit(clf(inputs), y)
        loss.backward()

        enc_opt.step()
        clf_opt.step()

    # Validation accuracy
    encoder.eval(); clf.eval()
    correct = total = 0
    with torch.no_grad():
        for batch in valid_loader:
            batch      = [t.float().to(device) for t in batch]
            views_orig = batch[:len(views)]
            y          = batch[2 * len(views)].long()
            hiddens, projs = encoder(*views_orig)
            inputs = projs if args.feature == 'latent' else hiddens
            preds  = clf(inputs).argmax(dim=1)
            correct += (preds == y).sum().item()
            total   += y.size(0)
    val_acc = correct / total

    enc_sch.step(val_acc)
    clf_sch.step(val_acc)

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_enc_sd  = {k: v.cpu().clone() for k, v in encoder.state_dict().items()}
        best_clf_sd  = {k: v.cpu().clone() for k, v in clf.state_dict().items()}
        patience_cnt = 0
    else:
        patience_cnt += 1

    if epoch % 10 == 0 or patience_cnt == 0:
        print(f'  Epoch {epoch:3d}  val_acc={val_acc:.4f}  best={best_val_acc:.4f}', flush=True)

    if patience_cnt >= patience_limit:
        print(f'  Early stop at epoch {epoch}')
        break

# Restore best weights
encoder.load_state_dict({k: v.to(device) for k, v in best_enc_sd.items()})
clf.load_state_dict({k: v.to(device) for k, v in best_clf_sd.items()})


# ── Masked evaluation ─────────────────────────────────────────────────────────
def _clf_forward_masked(clf, xs, mask_idx=None):
    """Classifier forward with view mask_idx zeroed out (None = no masking)."""
    xs = [torch.nan_to_num(x) for x in xs]

    if clf.args.feature == 'latent':
        stacked  = torch.stack(xs, dim=1)
        attended = clf.self_attention(stacked)[0] + stacked
        zs = [attended[:, i, :] for i in range(clf.num_views)]
    else:
        interacted = clf.interaction_layer(*xs)
        zs = [
            clf.output_layers[i](torch.cat([
                _pool(xs[i],         clf._use_last[i]),
                _pool(interacted[i], clf._use_last[i]),
            ], dim=-1))
            for i in range(clf.num_views)
        ]

    if mask_idx is not None:
        zs[mask_idx] = torch.zeros_like(zs[mask_idx])

    return clf.fc(torch.cat(zs, dim=-1))


def evaluate_masked(encoder, clf, loader, mask_idx=None):
    """Return accuracy with one view zeroed out (mask_idx=None → all views active)."""
    encoder.eval(); clf.eval()
    correct = total = 0
    with torch.no_grad():
        for batch in loader:
            batch      = [t.float().to(device) for t in batch]
            views_orig = batch[:len(views)]
            y          = batch[2 * len(views)].long()
            hiddens, projs = encoder(*views_orig)
            inputs = projs if clf.args.feature == 'latent' else hiddens
            logits = _clf_forward_masked(clf, inputs, mask_idx=mask_idx)
            preds  = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total   += y.size(0)
    return correct / total


# ── Results ───────────────────────────────────────────────────────────────────
print('\n' + '=' * 60)
print('VIEW MASKING RESULTS (test set)')
print('=' * 60)
print(f'  Views: {list(views)}')
print(f'  Pretrain data: {args.pretrain_data_name}')
print(f'  Finetune data: {args.data_name}')
print()

full_acc = evaluate_masked(encoder, clf, test_loader, mask_idx=None)
print(f'  All views active:  acc={full_acc:.4f}')
print()

for i, view_name in enumerate(views):
    masked_acc = evaluate_masked(encoder, clf, test_loader, mask_idx=i)
    drop = full_acc - masked_acc
    print(f'  Mask view {i} ({view_name:6s}): acc={masked_acc:.4f}  drop={drop:+.4f}')

print('=' * 60)
print()
print('Interpretation: larger drop = higher contribution from that view.')

"""
run_probe.py — Isolated evaluation of paper's classification head G.

The paper defines G as: G: cat([zt, zd, zf]) → Linear → logits
where zt/zd/zf are the projector outputs Ft/Fd/Ff from the pretrained encoder.

Two probe modes
---------------
raw        G applied to mean-pooled raw view features (no encoder at all).
           Baseline: how much does representation learning contribute?

pretrained G applied to frozen pretrained encoder projector outputs zt/zd/zf.
           Standard linear-probing evaluation of SSL representation quality.
           Transfer-learning setting: pretrained on source, probed on target.

Usage
-----
  python run_probe.py --data_name _DA_HAR70plus_256_00 \
      --pretrain_data_name _DA_HARTH_256_00 \
      --num_feature 6 --num_target 7 \
      --view2 dx --view3 xf --logsig_depth 2 \
      --epochs_pretrain 2 --epochs_finetune 50 --seed 0 \
      --probe_type pretrained

  python run_probe.py ... --probe_type raw
"""

import os
import sys
import random
import pickle
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.metrics import (precision_recall_fscore_support,
                             roc_auc_score, average_precision_score)
from sklearn.preprocessing import label_binarize

# Strip --probe_type from sys.argv before parse_args() sees it
_probe_type = 'pretrained'
_clean_argv = []
_skip_next = False
for _tok in sys.argv[1:]:
    if _skip_next:
        _probe_type = _tok
        _skip_next = False
        continue
    if _tok == '--probe_type':
        _skip_next = True
    elif _tok.startswith('--probe_type='):
        _probe_type = _tok.split('=', 1)[1]
    else:
        _clean_argv.append(_tok)
sys.argv = [sys.argv[0]] + _clean_argv

from src.config import parse_args
from src.dataloader import Load_Dataset, preprocess_data, get_view_num_features
from src.model import Encoder
from src.trainer import load_encoder
from src.evaluation import repeat_if_batch_size_one

use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')

# ---------------------------------------------------------------------------
# Args & seeding
# ---------------------------------------------------------------------------

args = parse_args()
args.probe_type = _probe_type

def seed_everything(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

seed_everything(args.seed)

if args.pretrain_data_name is None:
    args.pretrain_data_name = args.data_name

args.context_len = int(args.data_name.split('_')[3])
args.horizon_len = int(args.data_name.split('_')[4])

if args.num_feature > 64:
    args.num_feature = 64

args.num_feature_v2 = get_view_num_features(args.view2, args.num_feature, args.logsig_depth)
args.num_feature_v3 = get_view_num_features(args.view3, args.num_feature, args.logsig_depth)

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

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

views = ('xt', args.view2, args.view3)
_logsig_kw = dict(
    logsig_depth=args.logsig_depth,
    logsig_mode=getattr(args, 'logsig_mode', 'stream'),
    logsig_window_size=getattr(args, 'logsig_window_size', 32),
    logsig_smoothing=getattr(args, 'logsig_smoothing', 'tukey'),
    logsig_smooth_param=getattr(args, 'logsig_smooth_param', 0.5),
)

pre_tv = preprocess_data(X_tr, X_va, views=views, **_logsig_kw)
pre_tt = preprocess_data(X_tr, X_te, views=views, **_logsig_kw)

Xtr1, Xva1 = pre_tv['v1'][0], pre_tv['v1'][1]
Xtr2, Xva2 = pre_tv['v2'][0], pre_tv['v2'][1]
Xtr3, Xva3 = pre_tv['v3'][0], pre_tv['v3'][1]
Xtr1, Xte1 = pre_tt['v1'][0], pre_tt['v1'][1]
Xtr2, Xte2 = pre_tt['v2'][0], pre_tt['v2'][1]
Xtr3, Xte3 = pre_tt['v3'][0], pre_tt['v3'][1]

def make_loader(v1, v2, v3, y, mode):
    ds = Load_Dataset([v1, v2, v3], [v1, v2, v3], y, mode, views=views)
    return DataLoader(ds, batch_size=args.batch_size_finetune,
                      shuffle=(mode == 'finetune'), drop_last=False,
                      num_workers=4, pin_memory=True, persistent_workers=True)

train_loader = make_loader(Xtr1, Xtr2, Xtr3, y_tr, 'finetune')
valid_loader = make_loader(Xva1, Xva2, Xva3, y_va, 'test')
test_loader  = make_loader(Xte1, Xte2, Xte3, y_te, 'test')

# ---------------------------------------------------------------------------
# Feature extractor
# ---------------------------------------------------------------------------

def pool_view(x: torch.Tensor, view_name: str) -> torch.Tensor:
    """[N, L, C] → [N, C]: last-step for logsig (cumulative), mean otherwise."""
    return x[:, -1, :] if view_name == 'logsig' else x.mean(dim=1)


if args.probe_type == 'raw':
    encoder = None
    feat_dim = args.num_feature + args.num_feature_v2 + args.num_feature_v3
    print(f'[probe=raw] feat_dim={feat_dim} '
          f'(xt:{args.num_feature} {args.view2}:{args.num_feature_v2} {args.view3}:{args.num_feature_v3})')

else:
    pretrain_tag = (f'{args.pretrain_data_name}_v2{args.view2}_v3{args.view3}'
                    f'_ep{args.epochs_pretrain}_{args.seed}')
    ckpt = f'model_pretrain/{args.pretrain_data_name}/{pretrain_tag}.pth'
    if not os.path.exists(ckpt):
        print(f'Checkpoint not found: {ckpt}')
        sys.exit(1)

    encoder = Encoder(args)
    encoder = load_encoder(encoder, ckpt, {
        'input_layer_t': args.num_feature,
        'input_layer_d': args.num_feature_v2,
        'input_layer_f': args.num_feature_v3,
    })
    encoder = encoder.to(device)
    for p in encoder.parameters():
        p.requires_grad_(False)
    encoder.eval()

    feat_dim = 3 * args.num_hidden   # cat([zt, zd, zf])
    print(f'[probe=pretrained] feat_dim={feat_dim}, ckpt={ckpt}')


def get_features(Xt, dX, Xf) -> torch.Tensor:
    """Extract [N, feat_dim] features for a batch."""
    if encoder is None:
        return torch.cat([
            pool_view(Xt, 'xt'),
            pool_view(dX, args.view2),
            pool_view(Xf, args.view3),
        ], dim=-1)
    with torch.no_grad():
        _, _, _, zt, zd, zf = encoder(Xt, dX, Xf)
    return torch.cat([zt, zd, zf], dim=-1)


# ---------------------------------------------------------------------------
# G: paper's classification head — single linear layer
# ---------------------------------------------------------------------------

G = nn.Linear(feat_dim, args.num_target).to(device)
nn.init.normal_(G.weight, std=0.01)
nn.init.zeros_(G.bias)

optimizer = torch.optim.Adam(G.parameters(), lr=args.lr, weight_decay=args.weight_decay)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=10)
criterion = nn.CrossEntropyLoss()

# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def run_epoch(loader, train: bool):
    G.train() if train else G.eval()
    total_loss, correct, n = 0.0, 0, 0
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for batch in loader:
            batch = [repeat_if_batch_size_one(t.to(device)) for t in batch]
            Xt, dX, Xf, y = batch[0], batch[1], batch[2], batch[6]
            logits = G(get_features(Xt, dX, Xf))
            loss = criterion(logits, y)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * len(y)
            correct += (logits.detach().argmax(1) == y).sum().item()
            n += len(y)
    return total_loss / n, correct / n


@torch.no_grad()
def evaluate(loader) -> dict:
    G.eval()
    all_logits, all_y = [], []
    for batch in loader:
        batch = [repeat_if_batch_size_one(t.to(device)) for t in batch]
        Xt, dX, Xf, y = batch[0], batch[1], batch[2], batch[6]
        all_logits.append(G(get_features(Xt, dX, Xf)))
        all_y.append(y)
    logits = torch.cat(all_logits)
    y_true = torch.cat(all_y).cpu().numpy()
    y_pred = logits.argmax(1).cpu().numpy()
    y_score = torch.softmax(logits, dim=1).cpu().numpy()

    acc = (y_true == y_pred).mean()
    _, _, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted', zero_division=0)
    auroc = auprc = None
    if len(np.unique(y_true)) == args.num_target:
        if args.num_target == 2:
            auroc = roc_auc_score(y_true, y_score[:, 1])
            auprc = average_precision_score(y_true, y_score[:, 1])
        else:
            y_bin = label_binarize(y_true, classes=range(args.num_target))
            try:
                auroc = roc_auc_score(y_bin, y_score, average='macro', multi_class='ovr')
                auprc = average_precision_score(y_bin, y_score, average='macro')
            except Exception:
                pass
    return {'accuracy': acc, 'f1': f1, 'auroc': auroc, 'auprc': auprc}


best_val_acc = -1.0
best_state = None

for epoch in range(1, args.epochs_finetune + 1):
    tr_loss, tr_acc = run_epoch(train_loader, train=True)
    _, val_acc = run_epoch(valid_loader, train=False)
    scheduler.step(val_acc)
    if val_acc >= best_val_acc:
        best_val_acc = val_acc
        best_state = {k: v.clone() for k, v in G.state_dict().items()}
    if epoch % 10 == 0 or epoch == 1:
        print(f'Epoch {epoch:3d}: train_loss={tr_loss:.4f} train_acc={tr_acc:.4f} val_acc={val_acc:.4f}')

G.load_state_dict(best_state)
test_m = evaluate(test_loader)
print(f'\nTest — acc={test_m["accuracy"]:.4f}  f1={test_m["f1"]:.4f}  '
      f'auroc={test_m["auroc"]}  auprc={test_m["auprc"]}')

# ---------------------------------------------------------------------------
# Write to final_test_metric_summary.tsv
# ---------------------------------------------------------------------------

os.makedirs(f'out_finetune/{args.data_name}', exist_ok=True)
summary_file = f'out_finetune/{args.data_name}/final_test_metric_summary.tsv'

if args.probe_type == 'raw':
    run_name = (f'probe_raw_{args.data_name}'
                f'_v2{args.view2}_v3{args.view3}_seed{args.seed}')
else:
    pretrain_tag = (f'{args.pretrain_data_name}_v2{args.view2}_v3{args.view3}'
                    f'_ep{args.epochs_pretrain}_{args.seed}')
    run_name = f'probe_pt_{args.data_name}_from_{pretrain_tag}'

if not os.path.exists(summary_file):
    with open(summary_file, 'w') as f:
        f.write('run_name\tfinal_test_score\tepochs_trained\n')
with open(summary_file, 'a') as f:
    f.write(f'{run_name}\t{test_m["accuracy"]:.6f}\t{args.epochs_finetune}\n')

print(f'Summary written: {run_name}  score={test_m["accuracy"]:.4f}')

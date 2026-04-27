"""
inspect_attention.py — Visualise InteractionLayer attention weights.

For each InteractionLayer (encoder's and classifier's), computes the mean
3×3 attention matrix over a dataset: rows = query view, cols = key view.
A near-zero column means that view is ignored by all others.
A diagonal-dominant matrix means views only attend to themselves (no interaction).

Usage:
  python inspect_attention.py \
      --data_name _DA_HAR70plus_256_00 \
      --pretrain_data_name _DA_HARTH_256_00 \
      --num_feature 6 --num_target 7 \
      --view2 logsig --view3 xf --logsig_depth 2 \
      --epochs_pretrain 2 --seed 0
"""

import os, sys, pickle, random
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ---------------------------------------------------------------------------
# Strip unknown args before parse_args sees them (--split, etc.)
# ---------------------------------------------------------------------------
from src.config import parse_args
from src.dataloader import Load_Dataset, preprocess_data, get_view_num_features
from src.model import Encoder, Classifier
from src.trainer import load_encoder
from torch.utils.data import DataLoader

use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')

# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

args = parse_args()

if args.pretrain_data_name is None:
    args.pretrain_data_name = args.data_name

args.context_len = int(args.data_name.split('_')[3])
args.horizon_len = int(args.data_name.split('_')[4])

if args.num_feature > 64:
    args.num_feature = 64

args.num_feature_v2 = get_view_num_features(args.view2, args.num_feature, args.logsig_depth)
args.num_feature_v3 = get_view_num_features(args.view3, args.num_feature, args.logsig_depth)

# ---------------------------------------------------------------------------
# Data (use test split for inspection)
# ---------------------------------------------------------------------------

with open(f'preprocessed_data/{args.data_name}.pkl', 'rb') as f:
    (_, _, _, _,
     _, _, _, _,
     X_te_raw, _, _, y_test) = pickle.load(f)

X_te = torch.tensor(X_te_raw).transpose(1, 2).float()
y_te = torch.tensor(y_test)

views = ('xt', args.view2, args.view3)
_logsig_kw = dict(
    logsig_depth=args.logsig_depth,
    logsig_mode=getattr(args, 'logsig_mode', 'stream'),
    logsig_window_size=getattr(args, 'logsig_window_size', 32),
    logsig_smoothing=getattr(args, 'logsig_smoothing', 'tukey'),
    logsig_smooth_param=getattr(args, 'logsig_smooth_param', 0.5),
)
pre = preprocess_data(X_te, X_te, views=views, **_logsig_kw)
Xte1, Xte2, Xte3 = pre['v1'][0], pre['v2'][0], pre['v3'][0]

loader = DataLoader(
    Load_Dataset([Xte1, Xte2, Xte3], [Xte1, Xte2, Xte3], y_te, 'test', views=views),
    batch_size=64, shuffle=False, drop_last=False, num_workers=2)

# ---------------------------------------------------------------------------
# Load pretrained encoder
# ---------------------------------------------------------------------------

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
encoder = encoder.to(device).eval()

# Dummy classifier to get its interaction layer (randomly initialised weights —
# shows the untrained head's attention pattern for reference)
clf = Classifier(args).to(device).eval()

VIEW_NAMES = ['xt (temporal)', f'{args.view2} (view2)', f'{args.view3} (view3)']

# ---------------------------------------------------------------------------
# Collect attention weights
# ---------------------------------------------------------------------------

enc_attn_sum  = torch.zeros(3, 3)   # encoder InteractionLayer
clf_attn_sum  = torch.zeros(3, 3)   # classifier InteractionLayer
n_batches = 0

with torch.no_grad():
    for batch in loader:
        Xt, dX, Xf = batch[0].to(device), batch[1].to(device), batch[2].to(device)
        Xt = torch.nan_to_num(Xt)
        dX = torch.nan_to_num(dX)
        Xf = torch.nan_to_num(Xf)

        # --- encoder forward, manually to capture attn weights ---
        ht = encoder.transformer_encoder_t(
             encoder.positional_encoding(encoder.input_layer_t(Xt)))
        hd = encoder.transformer_encoder_d(
             encoder.positional_encoding(encoder.input_layer_d(dX)))
        hf = encoder.transformer_encoder_f(
             encoder.positional_encoding(encoder.input_layer_f(Xf)))

        _, _, _, enc_attn = encoder.interaction_layer(ht, hd, hf, return_attn=True)
        # enc_attn: [N, L, 3, 3] — mean over batch and time
        enc_attn_sum += enc_attn.mean(dim=(0, 1)).cpu()

        # --- classifier interaction layer (on same hidden states) ---
        _, _, _, clf_attn = clf.interaction_layer(ht, hd, hf, return_attn=True)
        clf_attn_sum += clf_attn.mean(dim=(0, 1)).cpu()

        n_batches += 1

enc_attn_mean = (enc_attn_sum / n_batches).numpy()
clf_attn_mean = (clf_attn_sum / n_batches).numpy()

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
fig.suptitle(
    f'Mean InteractionLayer attention weights\n'
    f'{args.pretrain_data_name} → {args.data_name}  |  '
    f'views: xt / {args.view2} / {args.view3}',
    fontsize=11, fontweight='bold'
)

short_names = ['xt', args.view2, args.view3]

for ax, matrix, label in [
    (axes[0], enc_attn_mean, 'Encoder InteractionLayer\n(pretrained)'),
    (axes[1], clf_attn_mean, 'Classifier InteractionLayer\n(randomly initialised)'),
]:
    im = ax.imshow(matrix, vmin=0, vmax=1, cmap='Blues')
    ax.set_title(label, fontsize=10)
    ax.set_xticks(range(3))
    ax.set_yticks(range(3))
    ax.set_xticklabels(short_names, fontsize=9)
    ax.set_yticklabels(short_names, fontsize=9)
    ax.set_xlabel('Key (attends to →)', fontsize=8)
    ax.set_ylabel('Query (← attended by)', fontsize=8)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f'{matrix[i, j]:.3f}',
                    ha='center', va='center',
                    color='white' if matrix[i, j] > 0.6 else 'black',
                    fontsize=10, fontweight='bold')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

plt.tight_layout()
out = f'attention_{args.pretrain_data_name}_v2{args.view2}_v3{args.view3}.png'
plt.savefig(out, dpi=150, bbox_inches='tight')
print(f'Saved: {out}')
plt.show()

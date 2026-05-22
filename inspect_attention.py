"""
inspect_attention.py — Visualise encoder attention weights.

Two modes:

1. Single checkpoint (default — uses best model):
   - T×T temporal self-attention heatmaps for each transformer view
   - InteractionLayer 3×3 cross-view attention matrix

2. Epoch evolution (--epoch_ckpt_dir DIR):
   - Loads epoch checkpoints saved by run_pretrain.py --save_every N
   - Shows how InteractionLayer attention and temporal locality evolve
   - Checkpoints expected: DIR/{run_tag}_ep{N}.pth

Usage:
  # Single checkpoint
  python inspect_attention.py \\
      --data_name _DA_HAR70plus_256_00 --pretrain_data_name _DA_HARTH_256_00 \\
      --num_feature 6 --num_target 7 \\
      --view2 logsig --view3 xf --logsig_depth 2 \\
      --epochs_pretrain 2 --seed 0

  # Epoch evolution (requires --save_every N during pretraining)
  python inspect_attention.py ... \\
      --epoch_ckpt_dir model_pretrain/_DA_HARTH_256_00/epoch_ckpts
"""

import os
import sys
import re
import glob
import argparse
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from sklearn.decomposition import PCA

from src.config import parse_args
from src.dataloader import preprocess_data, get_view_num_features, Load_Dataset
from src.model import Encoder
from src.trainer import load_encoder
from torch.utils.data import DataLoader

use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')

PLOT_DIR = 'attention_plots'
os.makedirs(PLOT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Extra CLI args (not in parse_args)
# ---------------------------------------------------------------------------

_extra = argparse.ArgumentParser(add_help=False)
_extra.add_argument('--epoch_ckpt_dir', default=None,
                    help='Directory of epoch checkpoints for evolution mode')
_extra.add_argument('--ckpt', default=None,
                    help='Explicit checkpoint path (overrides the auto-derived BEST_CKPT)')
_extra.add_argument('--n_samples', default=256, type=int,
                    help='Max samples to use for attention averaging')
_extra_args, _remaining = _extra.parse_known_args()
sys.argv = [sys.argv[0]] + _remaining
args = parse_args()

if args.pretrain_data_name is None:
    args.pretrain_data_name = args.data_name
if args.num_feature > 64:
    args.num_feature = 64
_gt = getattr(args, 'logsig_global_time', False)
args.num_feature_v2 = get_view_num_features(args.view2, args.num_feature, args.logsig_depth, _gt)
args.num_feature_v3 = get_view_num_features(args.view3, args.num_feature, args.logsig_depth, _gt)

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

with open(f'preprocessed_data/{args.data_name}.pkl', 'rb') as f:
    (X_tr_raw, _, _, y_train,
     X_va_raw, _, _, y_val,
     X_te_raw, _, _, y_test) = pickle.load(f)

for X_raw, y_raw, split in [(X_te_raw, y_test, 'test'),
                              (X_va_raw, y_val, 'val'),
                              (X_tr_raw, y_train, 'train')]:
    if len(X_raw) > 0:
        print(f'Using {split} split  ({len(X_raw)} samples)', flush=True)
        break

X_te = torch.tensor(X_raw).transpose(1, 2).float()
y_te = torch.tensor(y_raw)

# Subsample for speed
n = min(_extra_args.n_samples, len(X_te))
idx = torch.randperm(len(X_te))[:n]
X_te, y_te = X_te[idx], y_te[idx]

views = ('xt', args.view2, args.view3)
_logsig_kw = dict(
    logsig_depth=args.logsig_depth,
    logsig_mode=getattr(args, 'logsig_mode', 'stream'),
    logsig_window_size=getattr(args, 'logsig_window_size', 32),
    logsig_smoothing=getattr(args, 'logsig_smoothing', 'tukey'),
    logsig_smooth_param=getattr(args, 'logsig_smooth_param', 0.5),
    logsig_stride=getattr(args, 'logsig_stride', 1),
    logsig_global_time=_gt,
)
pre = preprocess_data(X_te, X_te, views=views, **_logsig_kw)
Xte1, Xte2, Xte3 = pre['v1'][0], pre['v2'][0], pre['v3'][0]

loader = DataLoader(
    Load_Dataset([Xte1, Xte2, Xte3], [Xte1, Xte2, Xte3], y_te, 'test', views=views),
    batch_size=64, shuffle=False, drop_last=False, num_workers=2)

# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

_enc_suffix = f'_{args.encoder_type}' if args.encoder_type != 'transformer' else ''
_ls_mode = getattr(args, 'logsig_mode', 'stream')
_ls_wsiz = getattr(args, 'logsig_window_size', 32)
_ls_smooth = getattr(args, 'logsig_smoothing', 'tukey')
_ls_stride = getattr(args, 'logsig_stride', 1)
_ls_pool = getattr(args, 'logsig_pool', 'auto')

def _lsig_suffix():
    if _ls_mode == 'stream':
        base = ''
    elif _ls_mode == 'window':
        base = f'_win{_ls_wsiz}'
    else:
        base = f'_{_ls_smooth}{_ls_wsiz}'
    if _ls_stride > 1:
        base += f'_s{_ls_stride}'
    if _ls_pool != 'auto':
        base += f'_p{_ls_pool}'
    return base

_il_suffix = ('' if getattr(args, 'interaction_type', 'attention') == 'attention'
              else f'_il{args.interaction_type.replace("_", "")}')
RUN_TAG = (f'{args.pretrain_data_name}_v2{args.view2}_v3{args.view3}'
           f'_ep{args.epochs_pretrain}_{args.seed}'
           f'{_enc_suffix}{_lsig_suffix()}{_il_suffix}')
BEST_CKPT = f'model_pretrain/{args.pretrain_data_name}/{RUN_TAG}.pth'


def _load_encoder(ckpt_path):
    enc = Encoder(args)
    # Only pass dimension checks for plain Linear input layers.
    # MLP-based layers (mlp_logsig) have nested keys like input_layer_d.net.0.weight
    # and must be loaded as-is.
    try:
        import torch as _torch
        _sd = _torch.load(ckpt_path, map_location='cpu', weights_only=False)
        if isinstance(_sd, dict) and 'encoder_state_dict' in _sd:
            _sd = _sd['encoder_state_dict']
    except Exception:
        _sd = {}
    _candidates = {
        'input_layer_t': args.num_feature,
        'input_layer_d': args.num_feature_v2,
        'input_layer_f': args.num_feature_v3,
    }
    _new_num_features = {k: v for k, v in _candidates.items()
                         if f'{k}.weight' in _sd}
    enc = load_encoder(enc, ckpt_path, _new_num_features or None)
    return enc.to(device).eval()


# ---------------------------------------------------------------------------
# Attention extraction
# ---------------------------------------------------------------------------

def _register_temporal_hooks(encoder):
    """Register hooks on all TransformerEncoderLayer.self_attn modules.

    Returns (hook_handles, captured_dict).
    captured_dict maps view_name → list of [N, T, T] tensors (one per layer).
    After a forward pass, call _collect_temporal(captured_dict, view_name) to
    get the layer-averaged [N, T, T] matrix.
    """
    captured = {}
    handles = []

    def _hook_for(view_name, layer_idx):
        active = [False]
        def hook(module, inp, out):
            if active[0]:
                return
            active[0] = True
            try:
                q = inp[0]          # [N, T, D]  (batch_first)
                with torch.no_grad():
                    _, w = module(q, q, q, need_weights=True, average_attn_weights=True)
                key = (view_name, layer_idx)
                captured[key] = w.detach().cpu()   # [N, T, T]
            finally:
                active[0] = False
        return hook

    for view_name, attr in [('xt', 'transformer_encoder_t'),
                             ('v2',  'transformer_encoder_d'),
                             ('v3',  'transformer_encoder_f')]:
        te = getattr(encoder, attr, None)
        if te is None:
            continue
        for li, layer in enumerate(te.layers):
            h = layer.self_attn.register_forward_hook(_hook_for(view_name, li))
            handles.append(h)

    return handles, captured


def _collect_temporal(captured, view_name):
    """Average attention over layers → [N, T, T]."""
    keys = sorted(k for k in captured if k[0] == view_name)
    if not keys:
        return None
    return torch.stack([captured[k] for k in keys], dim=0).mean(dim=0)


def _run_inference(encoder):
    """Run the full dataset through encoder; return dicts of accumulated attention."""
    handles, captured = _register_temporal_hooks(encoder)

    cross_sum = torch.zeros(3, 3)
    temporal_sum = {'xt': None, 'v2': None, 'v3': None}

    # Per-head cross-view accumulator — only for standard InteractionLayer
    il = encoder.interaction_layer
    _has_per_head = hasattr(il, 'multihead_attn')
    num_heads = il.multihead_attn.num_heads if _has_per_head else 0
    cross_sum_per_head = torch.zeros(num_heads, 3, 3) if _has_per_head else None

    n_batches = 0

    with torch.no_grad():
        for batch in loader:
            Xt, dX, Xf = batch[0].to(device), batch[1].to(device), batch[2].to(device)
            encoder(Xt, dX, Xf)   # triggers hooks

            # Re-extract pre-interaction hidden states
            ht_h, hd_h, hf_h, _, _, _ = encoder(Xt, dX, Xf)
            N, L, D = ht_h.shape
            h = torch.stack([ht_h, hd_h, hf_h], dim=2).view(N * L, 3, D)

            if _has_per_head:
                _, w_ph = il.multihead_attn(
                    h, h, h, need_weights=True, average_attn_weights=False
                )  # [N*L, num_heads, 3, 3]
                if n_batches == 0:
                    print(f'DEBUG w_ph shape: {w_ph.shape}', flush=True)
                    print(f'DEBUG w_ph[0] (first token, all heads):\n{w_ph[0].cpu().numpy().round(4)}', flush=True)
                    print(f'DEBUG ht==hd: {torch.allclose(ht_h, hd_h, atol=1e-3)}  ht==hf: {torch.allclose(ht_h, hf_h, atol=1e-3)}', flush=True)
                cross_sum_per_head += w_ph.view(N, L, num_heads, 3, 3).mean(dim=(0, 1)).cpu()
                cross_sum += w_ph.mean(dim=1).view(N, L, 3, 3).mean(dim=(0, 1)).cpu()
            else:
                _, _, _, cw = il(ht_h, hd_h, hf_h, return_attn=True)
                cross_sum += cw.mean(dim=(0, 1)).cpu()

            # Temporal per view
            for vn in ('xt', 'v2', 'v3'):
                t = _collect_temporal(captured, vn)
                if t is not None:
                    if temporal_sum[vn] is None:
                        temporal_sum[vn] = t.mean(dim=0)  # [T, T]
                    else:
                        temporal_sum[vn] += t.mean(dim=0)

            n_batches += 1
            captured.clear()

    for h in handles:
        h.remove()

    cross_mean = (cross_sum / n_batches).numpy()
    cross_mean_per_head = (cross_sum_per_head / n_batches).numpy() if _has_per_head else None
    temporal_mean = {k: (v / n_batches).numpy() if v is not None else None
                     for k, v in temporal_sum.items()}
    return cross_mean, cross_mean_per_head, temporal_mean


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

VIEW_DISPLAY = {'xt': 'xt (temporal)', 'v2': args.view2, 'v3': args.view3}

def _plot_cross_view(ax, matrix, title, view_names=None):
    if view_names is None:
        view_names = ['xt', args.view2, args.view3]
    im = ax.imshow(matrix, vmin=0, vmax=1, cmap='Blues')
    ax.set_title(title, fontsize=9)
    ax.set_xticks(range(3)); ax.set_yticks(range(3))
    ax.set_xticklabels(view_names, fontsize=8)
    ax.set_yticklabels(view_names, fontsize=8)
    ax.set_xlabel('Key (attends to)', fontsize=7)
    ax.set_ylabel('Query', fontsize=7)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f'{matrix[i,j]:.4f}',
                    ha='center', va='center', fontsize=8, fontweight='bold',
                    color='white' if matrix[i, j] > 0.6 else 'black')
    return im


def _plot_temporal(ax, matrix, title):
    T = matrix.shape[0]
    ax.imshow(matrix, cmap='viridis', aspect='auto',
              vmin=0, vmax=matrix.max())
    ax.set_title(title, fontsize=9)
    ax.set_xlabel('Key position', fontsize=7)
    ax.set_ylabel('Query position', fontsize=7)
    step = max(1, T // 8)
    ticks = list(range(0, T, step))
    ax.set_xticks(ticks); ax.set_yticks(ticks)
    ax.tick_params(labelsize=6)


def _locality_score(matrix, bandwidth=16):
    """Fraction of attention within ±bandwidth of the diagonal."""
    T = matrix.shape[0]
    mask = np.abs(np.arange(T)[:, None] - np.arange(T)[None, :]) <= bandwidth
    return float((matrix * mask).sum() / matrix.sum())


# ---------------------------------------------------------------------------
# Investigation 1: pre-softmax logit analysis
# ---------------------------------------------------------------------------

def _extract_pre_softmax_logits(encoder, loader):
    """Manually compute Q·Kᵀ / sqrt(d_k) for the InteractionLayer MHA.

    Returns (all_logits, num_heads) where all_logits is [N_total*L, num_heads, 3, 3],
    or None if the interaction layer is not a standard InteractionLayer.
    """
    il = encoder.interaction_layer
    if not hasattr(il, 'multihead_attn'):
        print('Skipping logit analysis: interaction layer has no multihead_attn')
        return None

    mha = il.multihead_attn
    E = mha.embed_dim
    num_heads = mha.num_heads
    head_dim = E // num_heads

    W = mha.in_proj_weight.detach()          # [3E, E]
    b = mha.in_proj_bias.detach() if mha.in_proj_bias is not None else None
    W_q, W_k = W[:E], W[E:2*E]              # [E, E] each
    b_q = b[:E]   if b is not None else None
    b_k = b[E:2*E] if b is not None else None

    all_logits = []

    with torch.no_grad():
        for batch in loader:
            Xt, dX, Xf = batch[0].to(device), batch[1].to(device), batch[2].to(device)
            ht, hd, hf, _, _, _ = encoder(Xt, dX, Xf)
            N, L, D = ht.shape
            h = torch.stack([ht, hd, hf], dim=2).view(N * L, 3, D)  # [N*L, 3, D]

            Q = h @ W_q.T  # [N*L, 3, E]
            K = h @ W_k.T
            if b_q is not None:
                Q = Q + b_q
                K = K + b_k

            # Reshape to multi-head: [N*L, num_heads, 3, head_dim]
            Q = Q.view(N * L, 3, num_heads, head_dim).permute(0, 2, 1, 3)
            K = K.view(N * L, 3, num_heads, head_dim).permute(0, 2, 1, 3)

            logits = Q @ K.transpose(-2, -1) / (head_dim ** 0.5)  # [N*L, num_heads, 3, 3]
            all_logits.append(logits.cpu())

    return torch.cat(all_logits, dim=0), num_heads  # [N_total*L, num_heads, 3, 3]


def _plot_logit_analysis(all_logits, num_heads, view_names, run_tag):
    """Two-figure output: per-head mean logit matrices + logit range distribution."""
    # Per-row logit range: max - min over keys → [NT, num_heads, 3]
    logit_range = all_logits.max(dim=-1).values - all_logits.min(dim=-1).values

    colors_v = ['#4C72B0', '#DD8452', '#55A868']

    # Figure A: mean and std logit matrices per head
    fig_a, axes_a = plt.subplots(2, num_heads, figsize=(3.5 * num_heads, 7),
                                  squeeze=False)
    fig_a.suptitle(
        f'Pre-softmax Q·Kᵀ/√d logits — InteractionLayer\n'
        f'views: {" / ".join(view_names)}',
        fontsize=11, fontweight='bold'
    )
    for hi in range(num_heads):
        mean_mat = all_logits[:, hi].mean(dim=0).numpy()   # [3, 3]
        std_mat  = all_logits[:, hi].std(dim=0).numpy()

        for row, (mat, title, cmap) in enumerate([
            (mean_mat, f'Head {hi} — mean logit', 'RdBu_r'),
            (std_mat,  f'Head {hi} — std logit',  'Oranges'),
        ]):
            ax = axes_a[row, hi]
            vabs = max(abs(mat.min()), abs(mat.max())) if row == 0 else None
            im = ax.imshow(mat, cmap=cmap,
                           vmin=-vabs if row == 0 else 0,
                           vmax= vabs if row == 0 else None,
                           aspect='auto')
            ax.set_title(title, fontsize=9)
            ax.set_xticks(range(3)); ax.set_yticks(range(3))
            ax.set_xticklabels(view_names, fontsize=8)
            ax.set_yticklabels(view_names, fontsize=8)
            ax.set_xlabel('Key', fontsize=7); ax.set_ylabel('Query', fontsize=7)
            for i in range(3):
                for j in range(3):
                    ax.text(j, i, f'{mat[i,j]:.3f}', ha='center', va='center', fontsize=8,
                            color='white' if (row == 0 and abs(mat[i,j]) > 0.6 * vabs) else 'black')
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    out_a = os.path.join(PLOT_DIR, f'logits_{run_tag}_matrices.png')
    fig_a.savefig(out_a, dpi=150, bbox_inches='tight')
    print(f'Saved: {out_a}')

    # Figure B: logit range distribution (key diagnostic)
    fig_b, axes_b = plt.subplots(1, 2, figsize=(10, 4))
    fig_b.suptitle(
        'Logit range (max − min per query row)\n'
        'Near 0 → softmax cannot distinguish views → uniform attention',
        fontsize=10, fontweight='bold'
    )

    ax = axes_b[0]
    for qi, (qname, col) in enumerate(zip(view_names, colors_v)):
        ranges = logit_range[:, :, qi].numpy().flatten()
        rmin, rmax = float(ranges.min()), float(ranges.max())
        bins = 80 if rmax > rmin else 1
        ax.hist(ranges, bins=bins, alpha=0.55, label=f'query = {qname}',
                color=col, density=(rmax > rmin))
        ax.axvline(ranges.mean(), color=col, linestyle='--', linewidth=1.5)
    ax.set_xlabel('Logit range (max − min)', fontsize=9)
    ax.set_ylabel('Density', fontsize=9)
    ax.set_title('Distribution across all positions & heads', fontsize=9)
    ax.legend(fontsize=8, frameon=False)
    ax.spines[['top', 'right']].set_visible(False)

    ax = axes_b[1]
    head_means = logit_range.mean(dim=(0, 2)).numpy()   # [num_heads]
    head_stds  = logit_range.std(dim=(0, 2)).numpy()
    ax.bar(range(num_heads), head_means, yerr=head_stds,
           color='#4C72B0', alpha=0.8, capsize=5)
    ax.set_xticks(range(num_heads))
    ax.set_xticklabels([f'Head {i}' for i in range(num_heads)], fontsize=9)
    ax.set_ylabel('Mean logit range ± std', fontsize=9)
    ax.set_title('Per-head summary', fontsize=9)
    ax.spines[['top', 'right']].set_visible(False)

    plt.tight_layout()
    out_b = os.path.join(PLOT_DIR, f'logits_{run_tag}_range.png')
    fig_b.savefig(out_b, dpi=150, bbox_inches='tight')
    print(f'Saved: {out_b}')

    plt.close()


# ---------------------------------------------------------------------------
# Investigation 1b: bilinear score analysis (no multihead_attn → separate path)
# ---------------------------------------------------------------------------

def _extract_bilinear_scores(encoder, loader):
    """Extract pre-softmax bilinear scores and post-softmax attention.

    Returns (all_scores, all_attn): each [N_total*L, 3, 3].
    all_scores = h_i^T W_b[i,j] h_j / sqrt(D) before softmax.
    all_attn   = softmax(scores, dim=-1).
    Returns None if the interaction layer is not InteractionLayerBilinear.
    """
    il = encoder.interaction_layer
    if not hasattr(il, 'W_b'):
        print('Skipping bilinear score analysis: not an InteractionLayerBilinear')
        return None

    D = il.hidden_size
    W_b = il.W_b.detach()   # [3, 3, D, D]

    all_scores = []
    all_attn   = []

    with torch.no_grad():
        for batch in loader:
            Xt, dX, Xf = batch[0].to(device), batch[1].to(device), batch[2].to(device)
            ht, hd, hf, _, _, _ = encoder(Xt, dX, Xf)
            N, L, _ = ht.shape
            h = torch.stack([ht, hd, hf], dim=2).view(N * L, 3, D)  # [N*L, 3, D]

            scores = torch.einsum('nid,ijde,nje->nij', h, W_b, h) / (D ** 0.5)
            attn   = scores.softmax(dim=-1)

            all_scores.append(scores.cpu())
            all_attn.append(attn.cpu())

    return torch.cat(all_scores, dim=0), torch.cat(all_attn, dim=0)


def _plot_bilinear_analysis(all_scores, all_attn, view_names, run_tag):
    """Diagnostic figure for bilinear InteractionLayer.

    Figure A: score matrices (mean pre-softmax, std, mean post-softmax attn).
    Figure B: score range distribution — key diagnostic for discriminability.
    """
    colors_v = ['#4C72B0', '#DD8452', '#55A868']

    # Figure A: 3-panel matrix view
    fig_a, axes_a = plt.subplots(1, 3, figsize=(13, 4))
    fig_a.suptitle(
        f'Bilinear InteractionLayer — score & attention matrices\n'
        f'views: {" / ".join(view_names)}',
        fontsize=11, fontweight='bold'
    )

    mean_scores = all_scores.mean(dim=0).numpy()   # [3, 3]
    std_scores  = all_scores.std(dim=0).numpy()
    mean_attn   = all_attn.mean(dim=0).numpy()

    for ax, mat, title, cmap, center_zero in [
        (axes_a[0], mean_scores, 'Pre-softmax scores  h_i·W_b·h_j/√D\n(mean over all tokens)', 'RdBu_r', True),
        (axes_a[1], std_scores,  'Score std dev\n(high = discriminative)', 'Oranges', False),
        (axes_a[2], mean_attn,   'Post-softmax attention\n(mean over all tokens)', 'Blues', False),
    ]:
        if center_zero:
            vabs = max(abs(mat.min()), abs(mat.max())) + 1e-6
            im = ax.imshow(mat, cmap=cmap, vmin=-vabs, vmax=vabs, aspect='auto')
        else:
            im = ax.imshow(mat, cmap=cmap, vmin=0, vmax=None if not center_zero else 1,
                           aspect='auto')
        ax.set_title(title, fontsize=9)
        ax.set_xticks(range(3)); ax.set_yticks(range(3))
        ax.set_xticklabels(view_names, fontsize=8)
        ax.set_yticklabels(view_names, fontsize=8)
        ax.set_xlabel('Key (attends to)', fontsize=7)
        ax.set_ylabel('Query', fontsize=7)
        for i in range(3):
            for j in range(3):
                ax.text(j, i, f'{mat[i,j]:.4f}', ha='center', va='center', fontsize=8,
                        fontweight='bold',
                        color='white' if (not center_zero and mat[i,j] > 0.6 * mat.max() + 1e-6) else 'black')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    out_a = os.path.join(PLOT_DIR, f'bilinear_{run_tag}_matrices.png')
    fig_a.savefig(out_a, dpi=150, bbox_inches='tight')
    print(f'Saved: {out_a}')
    plt.close(fig_a)

    # Figure B: score range distribution (discriminability)
    score_range = all_scores.max(dim=-1).values - all_scores.min(dim=-1).values  # [NT, 3]

    fig_b, axes_b = plt.subplots(1, 3, figsize=(14, 4))
    fig_b.suptitle(
        'Bilinear score analysis — discriminability\n'
        'Score range near 0 → softmax is uniform → views not distinguished',
        fontsize=10, fontweight='bold'
    )

    # Panel 1: score range distribution per query
    ax = axes_b[0]
    for qi, (qname, col) in enumerate(zip(view_names, colors_v)):
        ranges = score_range[:, qi].numpy()
        _safe_hist(ax, ranges, alpha=0.55, label=f'query = {qname}  (mean={ranges.mean():.3f})',
                   color=col, density=True)
        ax.axvline(ranges.mean(), color=col, linestyle='--', linewidth=1.5)
    ax.set_xlabel('Score range per query row (max − min)', fontsize=9)
    ax.set_ylabel('Density', fontsize=9)
    ax.set_title('Score range distribution\n(higher = can route to specific views)', fontsize=9)
    ax.legend(fontsize=7, frameon=False)
    ax.spines[['top', 'right']].set_visible(False)

    # Panel 2: per-pair score distributions
    ax = axes_b[1]
    pair_labels = [f'{view_names[i]}→{view_names[j]}' for i in range(3) for j in range(3)]
    pair_colors = [colors_v[i] for i in range(3) for _ in range(3)]
    pair_styles = ['-', '--', ':'] * 3
    for idx, (i, j) in enumerate([(i, j) for i in range(3) for j in range(3)]):
        vals = all_scores[:, i, j].numpy()
        _safe_hist(ax, vals, alpha=0.35, label=pair_labels[idx],
                   color=pair_colors[idx], density=True,
                   histtype='step', linestyle=pair_styles[idx], linewidth=1.5)
    ax.axvline(0, color='black', linewidth=0.8, linestyle=':')
    ax.set_xlabel('Pre-softmax bilinear score', fontsize=9)
    ax.set_ylabel('Density', fontsize=9)
    ax.set_title('All 9 pair scores\n(separated = W_b learned routing)', fontsize=9)
    ax.legend(fontsize=6, frameon=False, ncol=2)
    ax.spines[['top', 'right']].set_visible(False)

    # Panel 3: bar chart of mean attn per entry (flattened 3×3)
    ax = axes_b[2]
    labels_flat = [f'{view_names[i]}→{view_names[j]}' for i in range(3) for j in range(3)]
    vals_flat   = mean_attn.flatten()
    bar_colors  = [colors_v[i] for i in range(3) for _ in range(3)]
    bar_alphas  = [1.0, 0.6, 0.3] * 3
    bars = ax.bar(range(9), vals_flat, color=bar_colors,
                  alpha=0.8, edgecolor='black', linewidth=0.5)
    ax.axhline(1/3, color='grey', linestyle='--', linewidth=1.2,
               label='uniform (1/3)')
    ax.set_xticks(range(9))
    ax.set_xticklabels(labels_flat, rotation=45, ha='right', fontsize=7)
    ax.set_ylabel('Mean post-softmax attention', fontsize=9)
    ax.set_title('Routing summary\n(deviation from 1/3 = learned preference)', fontsize=9)
    ax.legend(fontsize=8, frameon=False)
    ax.spines[['top', 'right']].set_visible(False)

    plt.tight_layout()
    out_b = os.path.join(PLOT_DIR, f'bilinear_{run_tag}_range.png')
    fig_b.savefig(out_b, dpi=150, bbox_inches='tight')
    print(f'Saved: {out_b}')
    plt.close(fig_b)


# ---------------------------------------------------------------------------
# Shared plotting helpers
# ---------------------------------------------------------------------------

def _safe_hist(ax, data, **kwargs):
    """ax.hist that degrades gracefully when data has zero or near-zero range."""
    lo, hi = float(data.min()), float(data.max())
    if hi <= lo:
        ax.axvline(lo, color=kwargs.get('color', 'gray'),
                   alpha=kwargs.get('alpha', 1.0), linewidth=2,
                   label=kwargs.get('label', ''))
        return
    try:
        ax.hist(data, bins=min(100, max(1, len(data) // 10)), **kwargs)
    except (ValueError, OverflowError):
        ax.axvline(float(data.mean()), color=kwargs.get('color', 'gray'),
                   alpha=kwargs.get('alpha', 1.0), linewidth=2,
                   label=kwargs.get('label', ''))


# ---------------------------------------------------------------------------
# Investigation 2: view embedding similarity audit
# ---------------------------------------------------------------------------

def _run_similarity_analysis(encoder, loader):
    """Compute pairwise cosine similarity between view embeddings before interaction.

    Returns (token_sims, sample_sims) — dicts keyed by 'xt-v2', 'xt-v3', 'v2-v3',
    each value a 1-D numpy array of per-token (or per-sample) cosine similarities.
    """
    token_sims   = {'xt-v2': [], 'xt-v3': [], 'v2-v3': []}
    sample_sims  = {'xt-v2': [], 'xt-v3': [], 'v2-v3': []}
    dot_products = {'xt-v2': [], 'xt-v3': [], 'v2-v3': []}
    norms        = {'xt': [], 'v2': [], 'v3': []}

    with torch.no_grad():
        for batch in loader:
            Xt, dX, Xf = batch[0].to(device), batch[1].to(device), batch[2].to(device)
            ht, hd, hf, _, _, _ = encoder(Xt, dX, Xf)
            N, L, D = ht.shape

            ht_f = ht.reshape(-1, D)
            hd_f = hd.reshape(-1, D)
            hf_f = hf.reshape(-1, D)

            token_sims['xt-v2'].append(F.cosine_similarity(ht_f, hd_f, dim=-1).cpu())
            token_sims['xt-v3'].append(F.cosine_similarity(ht_f, hf_f, dim=-1).cpu())
            token_sims['v2-v3'].append(F.cosine_similarity(hd_f, hf_f, dim=-1).cpu())

            dot_products['xt-v2'].append((ht_f * hd_f).sum(dim=-1).cpu())
            dot_products['xt-v3'].append((ht_f * hf_f).sum(dim=-1).cpu())
            dot_products['v2-v3'].append((hd_f * hf_f).sum(dim=-1).cpu())

            norms['xt'].append(ht_f.norm(dim=-1).cpu())
            norms['v2'].append(hd_f.norm(dim=-1).cpu())
            norms['v3'].append(hf_f.norm(dim=-1).cpu())

            ht_p = ht.mean(dim=1)
            hd_p = hd.mean(dim=1)
            hf_p = hf.mean(dim=1)

            sample_sims['xt-v2'].append(F.cosine_similarity(ht_p, hd_p, dim=-1).cpu())
            sample_sims['xt-v3'].append(F.cosine_similarity(ht_p, hf_p, dim=-1).cpu())
            sample_sims['v2-v3'].append(F.cosine_similarity(hd_p, hf_p, dim=-1).cpu())

    for k in token_sims:
        token_sims[k]  = torch.cat(token_sims[k]).numpy()
        sample_sims[k] = torch.cat(sample_sims[k]).numpy()
        dot_products[k] = torch.cat(dot_products[k]).numpy()
    for k in norms:
        norms[k] = torch.cat(norms[k]).numpy()

    return token_sims, sample_sims, dot_products, norms


def _plot_similarity_analysis(token_sims, sample_sims, view_names, run_tag):
    pair_keys    = ['xt-v2', 'xt-v3', 'v2-v3']
    pair_labels  = [f'{view_names[0]} – {view_names[1]}',
                    f'{view_names[0]} – {view_names[2]}',
                    f'{view_names[1]} – {view_names[2]}']
    colors       = ['#4C72B0', '#DD8452', '#55A868']

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    fig.suptitle(
        f'View embedding cosine similarity (pre-interaction layer)\n'
        f'views: {" / ".join(view_names)}',
        fontsize=11, fontweight='bold'
    )

    # Panel 1: token-level histogram
    ax = axes[0]
    for k, label, col in zip(pair_keys, pair_labels, colors):
        _safe_hist(ax, token_sims[k], alpha=0.5, label=label, color=col, density=True)
        ax.axvline(token_sims[k].mean(), color=col, linestyle='--', linewidth=1.5,
                   label=f'  mean={token_sims[k].mean():.3f}')
    ax.set_xlabel('Cosine similarity', fontsize=9)
    ax.set_ylabel('Density', fontsize=9)
    ax.set_title('Token-level (all positions)', fontsize=9)
    ax.legend(fontsize=7, frameon=False)
    ax.set_xlim(-1, 1)
    ax.spines[['top', 'right']].set_visible(False)

    # Panel 2: sample-level violin
    ax = axes[1]
    data = [sample_sims[k] for k in pair_keys]
    parts = ax.violinplot(data, positions=range(3), showmedians=True, showmeans=True)
    for pc, col in zip(parts['bodies'], colors):
        pc.set_facecolor(col); pc.set_alpha(0.6)
    ax.set_xticks(range(3))
    ax.set_xticklabels(pair_labels, fontsize=8)
    ax.set_ylabel('Cosine similarity', fontsize=9)
    ax.set_title('Sample-level (mean-pooled)', fontsize=9)
    ax.set_ylim(-1, 1)
    ax.axhline(0, color='grey', linewidth=0.8, linestyle=':')
    ax.spines[['top', 'right']].set_visible(False)

    # Panel 3: 3×3 mean cosine similarity matrix
    ax = axes[2]
    means = {k: token_sims[k].mean() for k in pair_keys}
    mat = np.ones((3, 3))
    mat[0, 1] = mat[1, 0] = means['xt-v2']
    mat[0, 2] = mat[2, 0] = means['xt-v3']
    mat[1, 2] = mat[2, 1] = means['v2-v3']
    im = ax.imshow(mat, vmin=-1, vmax=1, cmap='RdBu_r')
    ax.set_title('Mean token-level cosine similarity', fontsize=9)
    ax.set_xticks(range(3)); ax.set_yticks(range(3))
    ax.set_xticklabels(view_names, fontsize=8)
    ax.set_yticklabels(view_names, fontsize=8)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f'{mat[i,j]:.3f}', ha='center', va='center', fontsize=9,
                    fontweight='bold',
                    color='white' if abs(mat[i, j]) > 0.6 else 'black')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    out = os.path.join(PLOT_DIR, f'similarity_{run_tag}.png')
    fig.savefig(out, dpi=150, bbox_inches='tight')
    print(f'Saved: {out}')
    plt.close()


def _plot_dot_products(dot_products, norms, view_names, run_tag):
    """Raw dot products and vector norms — confirms whether near-zero cosine sim
    reflects true orthogonality (large norms, small dot products) or collapsed
    embeddings (near-zero norms making everything small)."""
    pair_keys   = ['xt-v2', 'xt-v3', 'v2-v3']
    pair_labels = [f'{view_names[0]} · {view_names[1]}',
                   f'{view_names[0]} · {view_names[2]}',
                   f'{view_names[1]} · {view_names[2]}']
    norm_keys   = ['xt', 'v2', 'v3']
    colors_pair = ['#4C72B0', '#DD8452', '#55A868']
    colors_view = ['#4C72B0', '#DD8452', '#55A868']

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    fig.suptitle(
        f'Raw dot products and embedding norms (pre-interaction layer)\n'
        f'views: {" / ".join(view_names)}',
        fontsize=11, fontweight='bold'
    )

    # Panel 1: dot product distributions
    ax = axes[0]
    for k, label, col in zip(pair_keys, pair_labels, colors_pair):
        _safe_hist(ax, dot_products[k], alpha=0.5, label=label, color=col, density=True)
        ax.axvline(dot_products[k].mean(), color=col, linestyle='--', linewidth=1.5,
                   label=f'  mean={dot_products[k].mean():.2f}')
    ax.axvline(0, color='black', linewidth=0.8, linestyle=':')
    ax.set_xlabel('Dot product  (ht · hd)', fontsize=9)
    ax.set_ylabel('Density', fontsize=9)
    ax.set_title('Raw dot products per token pair\n'
                 'Large norms + near-zero dot product = true orthogonality', fontsize=9)
    ax.legend(fontsize=7, frameon=False)
    ax.spines[['top', 'right']].set_visible(False)

    # Panel 2: norm distributions per view
    ax = axes[1]
    for k, col in zip(norm_keys, colors_view):
        label = view_names[norm_keys.index(k)]
        _safe_hist(ax, norms[k], alpha=0.5, label=f'{label}  (mean={norms[k].mean():.2f})',
                   color=col, density=True)
    ax.set_xlabel('L2 norm  ‖h‖', fontsize=9)
    ax.set_ylabel('Density', fontsize=9)
    ax.set_title('Embedding norms per view\n'
                 'Near-zero norm = collapsed encoder', fontsize=9)
    ax.legend(fontsize=8, frameon=False)
    ax.spines[['top', 'right']].set_visible(False)

    plt.tight_layout()
    out = os.path.join(PLOT_DIR, f'dotproducts_{run_tag}.png')
    fig.savefig(out, dpi=150, bbox_inches='tight')
    print(f'Saved: {out}')
    plt.close()


# ---------------------------------------------------------------------------
# PCA scatter of pre-interaction embeddings
# ---------------------------------------------------------------------------

def _plot_pca_scatter(encoder, loader, view_names, run_tag, max_tokens=2000):
    """3×3 cross-projection PCA scatter.

    Row i = PCA basis fitted on view i.
    Column j = which view is projected into that basis.
    Diagonal: a view in its own basis — spreads out.
    Off-diagonal: a foreign view in another's basis — collapses if orthogonal.
    """
    ht_all, hd_all, hf_all = [], [], []

    with torch.no_grad():
        for batch in loader:
            Xt, dX, Xf = batch[0].to(device), batch[1].to(device), batch[2].to(device)
            ht, hd, hf, _, _, _ = encoder(Xt, dX, Xf)
            N, L, D = ht.shape
            ht_all.append(ht.reshape(-1, D).cpu())
            hd_all.append(hd.reshape(-1, D).cpu())
            hf_all.append(hf.reshape(-1, D).cpu())

    embeddings = [
        torch.cat(ht_all).numpy(),
        torch.cat(hd_all).numpy(),
        torch.cat(hf_all).numpy(),
    ]

    rng = np.random.default_rng(0)
    n = min(max_tokens, len(embeddings[0]))
    idx = rng.choice(len(embeddings[0]), n, replace=False)
    embs = [e[idx] for e in embeddings]

    colors = ['#4C72B0', '#DD8452', '#55A868']
    fig, axes = plt.subplots(3, 3, figsize=(10, 9))
    fig.suptitle(
        f'Cross-projection PCA — pre-interaction embeddings\n'
        f'Row = PCA basis of that view   |   Col = view being projected\n'
        f'Diagonal: own basis (spread out)   Off-diagonal: foreign basis (collapses if orthogonal)',
        fontsize=9, fontweight='bold'
    )

    pcas = []
    for row, basis_emb in enumerate(embs):
        pca = PCA(n_components=2)
        pca.fit(basis_emb)
        pcas.append(pca)

        for col, proj_emb in enumerate(embs):
            ax = axes[row, col]
            coords = pca.transform(proj_emb)
            ax.scatter(coords[:, 0], coords[:, 1], s=3, alpha=0.25,
                       color=colors[col], rasterized=True)

            ev = pca.explained_variance_ratio_
            ax.set_xlabel(f'PC1 ({ev[0]*100:.1f}%)', fontsize=7)
            ax.set_ylabel(f'PC2 ({ev[1]*100:.1f}%)', fontsize=7)
            ax.tick_params(labelsize=6)
            ax.spines[['top', 'right']].set_visible(False)

            if col == 0:
                ax.set_ylabel(f'{view_names[row]} basis\nPC2 ({ev[1]*100:.1f}%)', fontsize=8)
            if row == 0:
                ax.set_title(f'project {view_names[col]}', fontsize=9, color=colors[col],
                             fontweight='bold')
            if row == col:
                for spine in ax.spines.values():
                    spine.set_visible(True)
                    spine.set_linewidth(1.5)
                    spine.set_edgecolor('#333333')

    plt.tight_layout()
    out = os.path.join(PLOT_DIR, f'pca_{run_tag}.png')
    fig.savefig(out, dpi=150, bbox_inches='tight')
    print(f'Saved: {out}')
    plt.close()


# ---------------------------------------------------------------------------
# Mode 1: single checkpoint
# ---------------------------------------------------------------------------

def run_single(ckpt_path):
    if not os.path.exists(ckpt_path):
        print(f'Checkpoint not found: {ckpt_path}')
        sys.exit(1)
    print(f'Loading: {ckpt_path}', flush=True)
    encoder = _load_encoder(ckpt_path)
    cross_mean, cross_mean_per_head, temporal_mean = _run_inference(encoder)

    transformer_views = [(k, v) for k, v in temporal_mean.items() if v is not None]
    n_temp = len(transformer_views)

    fig_cols = max(n_temp, 3)
    fig, axes = plt.subplots(2, fig_cols, figsize=(4 * fig_cols, 8))
    if fig_cols == 1:
        axes = axes.reshape(2, 1)

    fig.suptitle(
        f'Encoder attention — {args.pretrain_data_name} → {args.data_name}\n'
        f'views: xt / {args.view2} / {args.view3}',
        fontsize=11, fontweight='bold'
    )

    # Row 1: temporal attention heatmaps
    for col, (vn, mat) in enumerate(transformer_views):
        _plot_temporal(axes[0, col], mat, f'{VIEW_DISPLAY[vn]}\ntemporal self-attention')
    for col in range(len(transformer_views), fig_cols):
        axes[0, col].axis('off')

    # Row 2: cross-view attention + locality scores
    im = _plot_cross_view(axes[1, 0], cross_mean, 'InteractionLayer\ncross-view attention')
    plt.colorbar(im, ax=axes[1, 0], fraction=0.046, pad=0.04)

    # Locality bar chart
    ax_loc = axes[1, 1]
    loc_labels = [VIEW_DISPLAY[vn] for vn, _ in transformer_views]
    loc_scores = [_locality_score(mat) for _, mat in transformer_views]
    bars = ax_loc.bar(loc_labels, loc_scores, color='#4C72B0')
    ax_loc.set_ylim(0, 1)
    ax_loc.set_ylabel('Locality score\n(attention within ±16 steps)', fontsize=8)
    ax_loc.set_title('Temporal locality', fontsize=9)
    for b, s in zip(bars, loc_scores):
        ax_loc.text(b.get_x() + b.get_width() / 2, s + 0.02, f'{s:.2f}',
                    ha='center', fontsize=8)
    ax_loc.spines[['top', 'right']].set_visible(False)

    for col in range(2, fig_cols):
        axes[1, col].axis('off')

    plt.tight_layout()
    out = os.path.join(PLOT_DIR, f'attention_{RUN_TAG}.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    print(f'Saved: {out}')
    plt.close()

    # Investigation 1: pre-softmax logit analysis (standard IL) or bilinear score analysis
    result = _extract_pre_softmax_logits(encoder, loader)
    if result is not None:
        all_logits, n_heads = result
        _plot_logit_analysis(all_logits, n_heads,
                             ['xt', args.view2, args.view3], RUN_TAG)

    # Investigation 1b: bilinear score analysis
    bil_result = _extract_bilinear_scores(encoder, loader)
    if bil_result is not None:
        bil_scores, bil_attn = bil_result
        _plot_bilinear_analysis(bil_scores, bil_attn,
                                ['xt', args.view2, args.view3], RUN_TAG)

    # Investigation 2: view embedding similarity + dot products
    token_sims, sample_sims, dot_products, norms = _run_similarity_analysis(encoder, loader)
    _plot_similarity_analysis(token_sims, sample_sims,
                              ['xt', args.view2, args.view3], RUN_TAG)
    _plot_dot_products(dot_products, norms,
                       ['xt', args.view2, args.view3], RUN_TAG)

    # PCA scatter
    _plot_pca_scatter(encoder, loader, ['xt', args.view2, args.view3], RUN_TAG)

    # Per-head cross-view figure
    if cross_mean_per_head is not None:
        num_heads = cross_mean_per_head.shape[0]
        view_names = ['xt', args.view2, args.view3]
        fig2, axes2 = plt.subplots(1, num_heads, figsize=(3.5 * num_heads, 3.5))
        if num_heads == 1:
            axes2 = [axes2]
        fig2.suptitle(
            f'InteractionLayer — per-head cross-view attention\n'
            f'{args.pretrain_data_name} → {args.data_name}  |  views: xt / {args.view2} / {args.view3}',
            fontsize=10, fontweight='bold'
        )
        for head_idx, ax in enumerate(axes2):
            im = _plot_cross_view(ax, cross_mean_per_head[head_idx],
                                  f'Head {head_idx}', view_names)
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        plt.tight_layout()
        out2 = os.path.join(PLOT_DIR, f'attention_{RUN_TAG}_perhead.png')
        plt.savefig(out2, dpi=150, bbox_inches='tight')
        print(f'Saved: {out2}')
        plt.close()


# ---------------------------------------------------------------------------
# Mode 2: epoch evolution
# ---------------------------------------------------------------------------

def run_evolution(epoch_ckpt_dir):
    pattern = os.path.join(epoch_ckpt_dir, f'{RUN_TAG}_ep*.pth')
    ckpts = sorted(glob.glob(pattern),
                   key=lambda p: int(re.search(r'_ep(\d+)\.pth$', p).group(1)))
    if not ckpts:
        print(f'No epoch checkpoints found matching: {pattern}')
        sys.exit(1)
    print(f'Found {len(ckpts)} epoch checkpoints', flush=True)

    epochs, cross_matrices, locality = [], [], {v: [] for v in ('xt', 'v2', 'v3')}

    for ckpt_path in ckpts:
        ep = int(re.search(r'_ep(\d+)\.pth$', ckpt_path).group(1))
        print(f'  epoch {ep}: {ckpt_path}', flush=True)
        encoder = _load_encoder(ckpt_path)
        cross_mean, _, temporal_mean = _run_inference(encoder)
        epochs.append(ep)
        cross_matrices.append(cross_mean)
        for vn, mat in temporal_mean.items():
            if mat is not None:
                locality[vn].append(_locality_score(mat))
            else:
                locality[vn].append(None)

    n_ckpts = len(ckpts)
    view_names = ['xt', args.view2, args.view3]

    fig, axes = plt.subplots(2, n_ckpts, figsize=(3.5 * n_ckpts, 8))
    if n_ckpts == 1:
        axes = axes.reshape(2, 1)

    fig.suptitle(
        f'Attention evolution — {args.pretrain_data_name}\n'
        f'views: xt / {args.view2} / {args.view3}',
        fontsize=11, fontweight='bold'
    )

    # Row 1: cross-view attention per epoch
    for col, (ep, mat) in enumerate(zip(epochs, cross_matrices)):
        im = _plot_cross_view(axes[0, col], mat, f'Epoch {ep}', view_names)

    # Row 2: locality score over epochs (line plot spanning all columns)
    # Merge row-2 cells into a single axis by hiding all but first
    ax_ev = axes[1, 0]
    for col in range(1, n_ckpts):
        axes[1, col].axis('off')
    ax_ev.set_position([axes[1, 0].get_position().x0,
                        axes[1, 0].get_position().y0,
                        axes[1, n_ckpts - 1].get_position().x1 - axes[1, 0].get_position().x0,
                        axes[1, 0].get_position().height])

    colors = {'xt': '#4C72B0', 'v2': '#DD8452', 'v3': '#55A868'}
    for vn, vals in locality.items():
        valid = [(e, v) for e, v in zip(epochs, vals) if v is not None]
        if valid:
            ep_v, sc_v = zip(*valid)
            ax_ev.plot(ep_v, sc_v, marker='o', label=VIEW_DISPLAY[vn],
                       color=colors[vn])
    ax_ev.set_xlabel('Training epoch', fontsize=9)
    ax_ev.set_ylabel('Locality score (attention within ±16 steps)', fontsize=9)
    ax_ev.set_title('Temporal locality over training', fontsize=9)
    ax_ev.legend(fontsize=8, frameon=False)
    ax_ev.set_ylim(0, 1)
    ax_ev.spines[['top', 'right']].set_visible(False)

    plt.tight_layout()
    out = os.path.join(PLOT_DIR, f'attention_{RUN_TAG}_evolution.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    print(f'Saved: {out}')
    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if _extra_args.epoch_ckpt_dir:
    run_evolution(_extra_args.epoch_ckpt_dir)
else:
    run_single(_extra_args.ckpt if _extra_args.ckpt else BEST_CKPT)

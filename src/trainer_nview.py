"""trainer_nview.py — train/test functions for EncoderNView + ClassifierNView.

Parallel to src/trainer.py but works with any number of views.

The encoder's forward pass returns (hiddens: list, projections: list) rather
than the named ht/hd/hf/zt/zd/zf triple.  Contrastive loss sums NTXentLoss
over all views (equivalent to loss_type='ALL' from the 3-view original).

Expected batch format from Load_DatasetNView:
    (*views_orig[N], *views_aug[N], y)   — 2*N + 1 float tensors

Freeze mode unfreezes the input-projection weights of each branch:
  - _TransformerBranch: branches.i.proj.*
  - LogSigMLP:          branches.i.net.*
"""

import os
import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler
from pytorch_metric_learning import losses
from tqdm import tqdm


def _tqdm_disabled() -> bool:
    return os.environ.get('TQDM_DISABLE', '0').strip().lower() in {'1', 'true', 'yes', 'on'}


def _weight_reg(model, l2_scale=0.01):
    return sum(l2_scale * p.pow(2).sum() for p in model.parameters() if p.requires_grad)


def _is_input_layer(name: str) -> bool:
    """True for input-projection params inside EncoderNView.branches.

    Param names look like:  branches.<i>.<module>.<...>
    We unfreeze <module> in {'proj', 'net'} — the input-side weights.
    Transformer encoder weights live under branches.<i>.enc.* and are frozen.
    """
    parts = name.split('.')
    return len(parts) >= 3 and parts[0] == 'branches' and parts[2] in ('proj', 'net')


def train_nview(args, encoder, clf, encoder_optimizer, clf_optimizer,
                loader, mode='pretrain', device='cuda'):
    """Train one epoch.

    Modes:
        pretrain — encoder trains, clf ignored.
        finetune — encoder + clf both train.
        freeze   — encoder input projections + clf train; rest of encoder frozen.
    """
    num_views = encoder.num_views

    if mode == 'freeze':
        encoder.eval()
        for name, param in encoder.named_parameters():
            param.requires_grad = _is_input_layer(name)
    else:
        encoder.train()
        for param in encoder.parameters():
            param.requires_grad = True

    if mode != 'pretrain':
        clf.train()

    scaler   = GradScaler("cuda")
    info_loss = losses.NTXentLoss(temperature=args.temperature)
    info_crit = losses.SelfSupervisedLoss(info_loss, symmetric=True)
    cls_crit  = nn.CrossEntropyLoss()

    total_loss = total_loss_c = total_samples = 0

    pbar = tqdm(loader, desc=f"Training ({mode})", disable=_tqdm_disabled())
    for batch in pbar:
        batch      = [t.float().to(device) for t in batch]
        views_orig = batch[:num_views]
        views_aug  = batch[num_views: 2 * num_views]
        y          = batch[2 * num_views].long()

        encoder_optimizer.zero_grad()
        if mode != 'pretrain':
            clf_optimizer.zero_grad()

        with autocast("cuda", enabled=True):
            hiddens, projs         = encoder(*views_orig)
            hiddens_aug, projs_aug = encoder(*views_aug)

            contrastive = sum(info_crit(projs[i], projs_aug[i]) for i in range(num_views))
            loss = contrastive + _weight_reg(encoder)

            if mode != 'pretrain':
                inputs = projs if args.feature == 'latent' else hiddens
                logit  = clf(inputs)
                loss_c = cls_crit(logit, y)
                loss   = args.lam * loss + loss_c + _weight_reg(clf)

        scaler.scale(loss).backward()
        if mode != 'freeze':
            scaler.step(encoder_optimizer)
        if mode != 'pretrain':
            scaler.step(clf_optimizer)
        scaler.update()

        total_loss += loss.item() * y.size(0)
        if mode != 'pretrain':
            total_loss_c += loss_c.item() * y.size(0)
        total_samples += y.size(0)
        pbar.set_postfix({'loss': loss.item()})

    if mode == 'pretrain':
        return total_loss / total_samples
    return total_loss / total_samples, total_loss_c / total_samples


def test_nview(args, encoder, clf, loader, mode='pretrain', device='cuda'):
    """Evaluate one epoch (no gradient updates)."""
    num_views = encoder.num_views
    encoder.eval()
    if mode != 'pretrain':
        clf.eval()

    info_loss = losses.NTXentLoss(temperature=args.temperature)
    info_crit = losses.SelfSupervisedLoss(info_loss, symmetric=True)
    cls_crit  = nn.CrossEntropyLoss()

    total_loss = total_loss_c = total_samples = 0

    with torch.no_grad():
        pbar = tqdm(loader, desc=f"Testing ({mode})", disable=_tqdm_disabled())
        for batch in pbar:
            batch      = [t.float().to(device) for t in batch]
            views_orig = batch[:num_views]
            views_aug  = batch[num_views: 2 * num_views]
            y          = batch[2 * num_views].long()

            with autocast("cuda", enabled=True):
                hiddens, projs         = encoder(*views_orig)
                hiddens_aug, projs_aug = encoder(*views_aug)

                contrastive = sum(info_crit(projs[i], projs_aug[i]) for i in range(num_views))
                loss = contrastive + _weight_reg(encoder)

                if mode != 'pretrain':
                    inputs = projs if args.feature == 'latent' else hiddens
                    logit  = clf(inputs)
                    loss_c = cls_crit(logit, y)
                    loss   = args.lam * loss + loss_c + _weight_reg(clf)

            total_loss += loss.item() * y.size(0)
            if mode != 'pretrain':
                total_loss_c += loss_c.item() * y.size(0)
            total_samples += y.size(0)
            pbar.set_postfix({'loss': loss.item()})

    if mode == 'pretrain':
        return total_loss / total_samples
    return total_loss / total_samples, total_loss_c / total_samples

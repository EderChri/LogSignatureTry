import torch
import torch.nn as nn
import os
from torch.amp import autocast, GradScaler
from pytorch_metric_learning import losses
from tqdm import tqdm


def _tqdm_disabled() -> bool:
    return os.environ.get('TQDM_DISABLE', '0').strip().lower() in {'1', 'true', 'yes', 'on'}


def add_weight_regularization(model, l2_scale=0.01):
    return sum(l2_scale * p.pow(2).sum() for p in model.parameters() if p.requires_grad)


def _is_input_layer(name: str) -> bool:
    """True for input-projection params inside EncoderNView.branches.

    Param names: branches.<i>.<module>.<...>
    Unfreeze <module> in {'proj', 'net'} — the input-side weights.
    Transformer encoder weights live under branches.<i>.enc.* and are frozen.
    """
    parts = name.split('.')
    return len(parts) >= 3 and parts[0] == 'branches' and parts[2] in ('proj', 'net')


def train(args, encoder, clf, encoder_optimizer, clf_optimizer,
          loader, mode='pretrain', device='cuda'):
    """Train one epoch for any number of views.

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

    scaler    = GradScaler("cuda")
    info_loss = losses.NTXentLoss(temperature=args.temperature)
    info_crit = losses.SelfSupervisedLoss(info_loss, symmetric=True)
    cls_crit  = nn.CrossEntropyLoss()

    total_loss = total_loss_c = total_samples = 0

    pbar = tqdm(loader, desc=f"Training ({mode})", disable=_tqdm_disabled(), dynamic_ncols=True)
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
            loss = contrastive + add_weight_regularization(encoder)

            if mode != 'pretrain':
                inputs = projs if args.feature == 'latent' else hiddens
                logit  = clf(inputs)
                loss_c = cls_crit(logit, y)
                loss   = args.lam * loss + loss_c + add_weight_regularization(clf)

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


def test(args, encoder, clf, loader, mode='pretrain', device='cuda'):
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
        pbar = tqdm(loader, desc=f"Testing ({mode})", disable=_tqdm_disabled(), dynamic_ncols=True)
        for batch in pbar:
            batch      = [t.float().to(device) for t in batch]
            views_orig = batch[:num_views]
            views_aug  = batch[num_views: 2 * num_views]
            y          = batch[2 * num_views].long()

            with autocast("cuda", enabled=True):
                hiddens, projs         = encoder(*views_orig)
                hiddens_aug, projs_aug = encoder(*views_aug)

                contrastive = sum(info_crit(projs[i], projs_aug[i]) for i in range(num_views))
                loss = contrastive + add_weight_regularization(encoder)

                if mode != 'pretrain':
                    inputs = projs if args.feature == 'latent' else hiddens
                    logit  = clf(inputs)
                    loss_c = cls_crit(logit, y)
                    loss   = args.lam * loss + loss_c + add_weight_regularization(clf)

            total_loss += loss.item() * y.size(0)
            if mode != 'pretrain':
                total_loss_c += loss_c.item() * y.size(0)
            total_samples += y.size(0)
            pbar.set_postfix({'loss': loss.item()})

    if mode == 'pretrain':
        return total_loss / total_samples
    return total_loss / total_samples, total_loss_c / total_samples


# ---------------------------------------------------------------------------
# Pretrained model loader
# ---------------------------------------------------------------------------

def remove_module_prefix(state_dict):
    """Strip DataParallel ('module.') and torch.compile ('_orig_mod.') prefixes."""
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith('module.'):
            new_key = key[len('module.'):]
        elif key.startswith('_orig_mod.'):
            new_key = key[len('_orig_mod.'):]
        else:
            new_key = key
        new_state_dict[new_key] = value
    return new_state_dict


def load_encoder(encoder, checkpoint_path, new_num_features=None):
    """Load a pretrained encoder, re-initialising input layers when feature dims differ.

    Args:
        new_num_features: dict mapping layer name to expected input feature count,
                          e.g. {'input_layer_t': 6, 'input_layer_d': 21}.
                          Layers whose stored dim differs from the expected dim are
                          re-initialised with Xavier-uniform weights.
                          Pass None to skip dimension checking entirely.
    """
    try:
        state_dict = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    except TypeError:
        state_dict = torch.load(checkpoint_path, map_location='cpu')

    if isinstance(state_dict, dict) and 'encoder_state_dict' in state_dict:
        state_dict = state_dict['encoder_state_dict']

    state_dict = remove_module_prefix(state_dict)

    if new_num_features is not None:
        for layer_name, new_dim in new_num_features.items():
            old_weight = state_dict[f'{layer_name}.weight']
            old_bias   = state_dict[f'{layer_name}.bias']
            if new_dim != old_weight.size(1):
                new_weight = nn.Linear(new_dim, old_weight.size(0)).weight.data
                nn.init.xavier_uniform_(new_weight)
                state_dict[f'{layer_name}.weight'] = new_weight
                state_dict[f'{layer_name}.bias']   = torch.zeros_like(old_bias)

    for key, param in state_dict.items():
        if torch.is_tensor(param):
            if torch.isnan(param).any() or torch.isinf(param).any():
                state_dict[key] = torch.nan_to_num(param, nan=0.0, posinf=0.0, neginf=0.0)

    keys_to_remove = [key for key in state_dict.keys() if 'q_func' in key]
    for key in keys_to_remove:
        state_dict.pop(key)

    model_sd   = encoder.state_dict()
    state_dict = {k: v for k, v in state_dict.items()
                  if k not in model_sd or model_sd[k].shape == v.shape}

    encoder.load_state_dict(state_dict, strict=False)
    return encoder

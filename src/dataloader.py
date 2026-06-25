import fcntl
import os
import torch
import torch.fft as fft
import torchcde
from log_signatures_pytorch import log_signature, logsigdim
from torch.utils.data import Dataset
from typing import Tuple, Optional


def normalize(X_train: torch.Tensor, X_test: torch.Tensor, epsilon: float = 1e-8):
    mean = X_train.mean(dim=(0, 1), keepdim=True)
    std = X_train.std(dim=(0, 1), keepdim=True).clamp(min=epsilon)
    X_train_norm = (X_train - mean) / std
    X_test_norm = (X_test - mean) / std
    return X_train_norm, X_test_norm, mean, std


def add_time_feature(X: torch.Tensor):
    # X: [num_samples, sequence_length, num_features]
    num_samples, seq_length, _ = X.shape
    
    # Create a time index vector normalized between 0 and 1
    time_index = torch.linspace(0, 1, steps=seq_length).to(X.device)  # Shape: [sequence_length]
    
    # Expand time index to match the batch size
    time_feature = time_index.unsqueeze(0).unsqueeze(-1).repeat(num_samples, 1, 1)  # Shape: [num_samples, sequence_length, 1]
    
    # Concatenate the time feature to the original data
    X_with_time = torch.cat([time_feature, X], dim=-1)  # New shape: [num_samples, sequence_length, num_features + 1]
    return X_with_time


def get_dx(X: torch.Tensor) -> torch.Tensor:
    N, L, D = X.shape
    t = torch.linspace(0, 1, L, dtype=X.dtype, device=X.device)
    
    coeffs = torchcde.hermite_cubic_coefficients_with_backward_differences(X)
    spline = torchcde.CubicSpline(coeffs, t)
    dx = spline.derivative(t)
    return dx


def get_xf(X: torch.Tensor) -> torch.Tensor:
    return torch.abs(fft.fft(X.contiguous(), dim=1))


def _tukey_weights(W: int, alpha: float, dtype, device) -> torch.Tensor:
    """Tukey (tapered-cosine) window of length W, tapering ratio alpha.

    alpha=0 → rectangular, alpha=1 → Hann.
    Returns shape [W].
    """
    import math
    w = torch.ones(W, dtype=dtype, device=device)
    taper = int(math.floor(alpha * W / 2))
    if taper > 0:
        k = torch.arange(taper, dtype=dtype, device=device)
        cos_taper = 0.5 * (1 - torch.cos(math.pi * k / taper))
        w[:taper] = cos_taper
        w[W - taper:] = cos_taper.flip(0)
    return w


def _smooth_ema(X: torch.Tensor, alpha: float) -> torch.Tensor:
    """Exponential moving average along the time axis (dim=1)."""
    N, L, D = X.shape
    out = X.clone()
    for t in range(1, L):
        out[:, t, :] = alpha * X[:, t, :] + (1 - alpha) * out[:, t - 1, :]
    return out

def normalize_logsig_levels(
    logsig: torch.Tensor,
    depth: int,
    path_dim: int,
    num_copies: int = 1,
    global_time: bool = False,
    stats: list = None,
):
    """Normalize each level of the log-signature features to have zero mean and unit variance.

    Each truncation level gets one shared mean and std (scalar), computed over all
    samples, timesteps, and feature dimensions within that level.  For stacked
    multi-smooth signatures num_copies must equal K so that every copy is covered.

    Args:
        logsig:      [N, L, C] log-signature features.
        depth:       truncation depth of the log-signature.
        path_dim:    path dimensionality including the prepended time channel
                     (i.e. num_input_features + 1).
        num_copies:  number of stacked signature copies (K from multi_smooth_params).
        global_time: if True, the last channel is a global time feature and is
                     normalized with its own global z-score.
        stats:       list of (mean, std) tensors per level (output of a prior call
                     with stats=None).  When provided, these are applied instead of
                     computing fresh statistics from logsig — use this to apply
                     training-set statistics to the test split.
    Returns:
        (logsig_norm, stats): normalized [N, L, C] tensor and the list of
        (mean, std) pairs that were used, in level order (+ global_time entry last).
    """
    level_bounds = []
    prev = 0
    for k in range(1, depth + 1):
        end = logsigdim(path_dim, k)
        level_bounds.append((prev, end))
        prev = end
    C_sig = prev  # logsigdim(path_dim, depth)

    logsig_norm = logsig.clone()
    computed_stats = []
    stat_idx = 0
    for c in range(num_copies):
        base = c * C_sig
        for lvl_start, lvl_end in level_bounds:
            sl = slice(base + lvl_start, base + lvl_end)
            if stats is None:
                mean = logsig[..., sl].mean()
                std = logsig[..., sl].std().clamp(min=1e-8)
                computed_stats.append((mean, std))
            else:
                mean, std = stats[stat_idx]
                stat_idx += 1
            logsig_norm[..., sl] = (logsig[..., sl] - mean) / std

    if global_time:
        if stats is None:
            mean = logsig[..., -1].mean()
            std = logsig[..., -1].std().clamp(min=1e-8)
            computed_stats.append((mean, std))
        else:
            mean, std = stats[stat_idx]
        logsig_norm[..., -1] = (logsig[..., -1] - mean) / std

    return logsig_norm, computed_stats if stats is None else stats



def get_logsig(
    X: torch.Tensor,
    depth: int,
    mode: str = 'stream',
    window_size: int = 32,
    smoothing: str = 'tukey',
    smooth_param: float = 0.5,
    stride: int = 1,
    global_time: bool = False,
    multi_smooth_params: list = None,
    normalize: bool = True,
) -> torch.Tensor:
    """Log signature view of the time-augmented path.

    Three modes are supported:
    - 'stream':        Running log-sig of [0, t] at each step t.
                       Output shape: [N, L, C] (position 0 is zero-padded).
                       stride is ignored in stream mode.
    - 'window':        Log-sig of the sliding window [t-W+1, t] of size W.
    - 'window_smooth': Sliding window with smoothing applied to each window
                       segment before computing the signature.
                       smoothing='tukey': multiply window samples by a Tukey
                         (tapered-cosine) weight vector; smooth_param = alpha
                         tapering ratio (0 = rect, 1 = Hann).
                       smoothing='ema': replace each window with its EMA;
                         smooth_param = decay alpha.

    All modes prepend a t∈[0,1] time channel (local to each window/path) so
    that single-channel inputs still produce non-trivial signatures.

    Windowed modes const-pad early windows (before the window is full) to
    window_size:
      - mode='window' or smoothing='ema': replicate the first sample.
      - mode='window_smooth' + smoothing='tukey': zero-pad (matches the
        Tukey taper zeroing out the window edges).

    Args:
        X:            [N, L, D]
        depth:        truncation depth
        mode:         'stream' | 'window' | 'window_smooth'
        window_size:  sliding window length W (used for window modes)
        smoothing:    'tukey' | 'ema'  (used for window_smooth mode)
        smooth_param:        tapering ratio for tukey or alpha decay for ema
        stride:              compute a signature every stride steps; output length
                             is L // stride.  Ignored in stream mode.
        global_time:         if True, append the global position t/L as an extra
                             channel, giving the MLP branch a sense of absolute
                             position without a learned PE.
        multi_smooth_params: list of Tukey alpha floats.  When set and mode is
                             'window_smooth', computes one signature per alpha and
                             concatenates along the feature dim (K × C_sig).
                             Overrides smooth_param.  Ignored for other modes.

    Returns:
        stream mode:  [N, L,         C(+1)]  where C = logsigdim(D+1, depth)
        window modes: [N, L//stride, K*C(+1)] (K=1 when multi_smooth_params is None)
    """
    N, L, D = X.shape
    C_sig = logsigdim(D + 1, depth)
    _multi = (multi_smooth_params is not None
              and mode == 'window_smooth'
              and len(multi_smooth_params) > 0)
    K = len(multi_smooth_params) if _multi else 1
    C = K * C_sig + (1 if global_time else 0)

    if mode == 'stream':
        t = torch.linspace(0, 1, L, dtype=X.dtype, device=X.device)
        t = t.view(1, L, 1).expand(N, -1, -1)
        X_time = torch.cat([t, X], dim=-1)                    # [N, L, D+1]
        logsig = log_signature(X_time, depth, stream=True)    # [N, L-1, C_sig]
        pad = torch.zeros(N, 1, C_sig, dtype=X.dtype, device=X.device)
        result = torch.cat([pad, logsig], dim=1)              # [N, L, C_sig]
        if global_time:
            result = torch.cat([result, t], dim=-1)           # [N, L, C_sig+1]
        if normalize:
            result, _ = normalize_logsig_levels(result, depth, path_dim=D + 1, num_copies=1, global_time=global_time)
        return result

    # Windowed modes
    positions = range(stride, L + 1, stride) if stride > 1 else range(1, L + 1)
    L_out = len(range(stride, L + 1, stride)) if stride > 1 else L
    out = torch.zeros(N, L_out, C, dtype=X.dtype, device=X.device)

    for out_idx, end in enumerate(positions):
        start = max(0, end - window_size)
        seg = X[:, start:end, :].clone()                      # [N, W_real, D]

        # Const-pad early windows that haven't filled to window_size yet
        if seg.shape[1] < window_size:
            needed = window_size - seg.shape[1]
            if mode == 'window_smooth' and smoothing == 'tukey':
                pad_seg = torch.zeros(N, needed, D, dtype=X.dtype, device=X.device)
            else:
                pad_seg = seg[:, :1, :].expand(-1, needed, -1).clone()
            seg = torch.cat([pad_seg, seg], dim=1)

        W = seg.shape[1]  # == window_size after padding

        # Local time in [0, 1] within the window (shared across all param variants)
        seg_t = torch.linspace(0, 1, W, dtype=X.dtype, device=X.device)
        seg_t = seg_t.view(1, W, 1).expand(N, -1, -1)

        if _multi:
            parts = []
            for alpha in multi_smooth_params:
                seg_a = seg.clone()
                if smoothing == 'tukey':
                    weights = _tukey_weights(W, alpha, seg.dtype, seg.device)
                    seg_a = seg_a * weights.view(1, W, 1)
                else:
                    seg_a = _smooth_ema(seg_a, alpha)
                seg_full = torch.cat([seg_t, seg_a], dim=-1)
                parts.append(log_signature(seg_full, depth))  # [N, C_sig]
            sig_val = torch.cat(parts, dim=-1)                # [N, K*C_sig]
        else:
            if mode == 'window_smooth':
                if smoothing == 'tukey':
                    weights = _tukey_weights(W, smooth_param, seg.dtype, seg.device)
                    seg = seg * weights.view(1, W, 1)
                else:
                    seg = _smooth_ema(seg, smooth_param)
            seg_full = torch.cat([seg_t, seg], dim=-1)        # [N, W, D+1]
            sig_val = log_signature(seg_full, depth)          # [N, C_sig]

        if global_time:
            t_g = torch.full((N, 1), (end - 1) / max(L - 1, 1),
                             dtype=X.dtype, device=X.device)
            sig_val = torch.cat([sig_val, t_g], dim=-1)

        out[:, out_idx, :] = sig_val

    if normalize:
        out, _ = normalize_logsig_levels(out, depth, path_dim=D + 1, num_copies=K, global_time=global_time)
    return out


def get_view_num_features(view: str, num_feature: int, logsig_depth: int,
                          global_time: bool = False,
                          multi_smooth_params=None) -> int:
    """Input feature dimension produced by the given view transform."""
    if view in ('xt', 'dx', 'xf'):
        return num_feature
    elif view == 'logsig':
        c = logsigdim(num_feature + 1, logsig_depth)
        if multi_smooth_params is not None and len(multi_smooth_params) > 0:
            c *= len(multi_smooth_params)
        return c + 1 if global_time else c
    else:
        raise ValueError(f"Unknown view '{view}'. Choose from: xt, dx, xf, logsig")


def preprocess_data(X_train, X_test, views=('xt', 'dx', 'xf'), logsig_depth=2,
                    logsig_mode='stream', logsig_window_size=32,
                    logsig_smoothing='tukey', logsig_smooth_param=0.5,
                    logsig_stride=1, logsig_global_time=False,
                    logsig_multi_smooth_params=None,
                    logsig_normalize=False,
                    time_as_feature=False,
                    logsig_cache_key=None,
                    pca_components=None):
    """Preprocess training and test data for the requested views.

    Args:
        views:          tuple of three view names; first entry must be 'xt'.
        logsig_depth:   truncation depth used when a view is 'logsig'.
        pca_components: if set, reduce input channels to this many PCA components
                        (fit on training data) before computing any view.

    Returns:
        dict with keys 'v1', 'v2', 'v3', each a tuple
        (X_train, X_test, mean, std).
    """
    # Normalise time-domain data — used as input for all other transforms
    X_train_xt, X_test_xt, mean_xt, std_xt = normalize(X_train, X_test)

    # Optional PCA dimensionality reduction (fit on train, applied to train+test)
    if pca_components is not None and pca_components < X_train_xt.shape[-1]:
        from sklearn.decomposition import PCA as _PCA
        N_tr, L, D = X_train_xt.shape
        N_te = X_test_xt.shape[0]
        pca = _PCA(n_components=pca_components)
        pca.fit(X_train_xt.reshape(-1, D).numpy())
        X_train_xt = torch.from_numpy(
            pca.transform(X_train_xt.reshape(-1, D).numpy())
            .reshape(N_tr, L, pca_components)
        ).to(X_train.dtype)
        X_test_xt = torch.from_numpy(
            pca.transform(X_test_xt.reshape(-1, D).numpy())
            .reshape(N_te, L, pca_components)
        ).to(X_train.dtype)
        X_train_xt, X_test_xt, mean_xt, std_xt = normalize(X_train_xt, X_test_xt)
        print(f'[PCA] {D} → {pca_components} components', flush=True)

    results = {}
    for i, view in enumerate(views):
        key = f'v{i + 1}'
        if view == 'xt':
            data_tr, data_te = X_train_xt, X_test_xt
            mean, std = mean_xt, std_xt
            if time_as_feature:
                data_tr = add_time_feature(data_tr)
                data_te = add_time_feature(data_te)
        elif view == 'dx':
            data_tr = get_dx(X_train_xt)
            data_te = get_dx(X_test_xt)
            data_tr, data_te, mean, std = normalize(data_tr, data_te)
            if time_as_feature:
                data_tr = add_time_feature(data_tr)
                data_te = add_time_feature(data_te)
        elif view == 'xf':
            data_tr = get_xf(X_train_xt)
            data_te = get_xf(X_test_xt)
            data_tr, data_te, mean, std = normalize(data_tr, data_te)
            if time_as_feature:
                data_tr = add_time_feature(data_tr)
                data_te = add_time_feature(data_te)
        elif view == 'logsig':
            _logsig_kw = dict(
                depth=logsig_depth, mode=logsig_mode,
                window_size=logsig_window_size, smoothing=logsig_smoothing,
                smooth_param=logsig_smooth_param, stride=logsig_stride,
                global_time=logsig_global_time,
                multi_smooth_params=logsig_multi_smooth_params,
                normalize=False,  # per-level norm applied below using train stats
            )
            if logsig_cache_key:
                _cdir = 'preprocessed_data/.logsig_cache'
                os.makedirs(_cdir, exist_ok=True)
                _ctr = f'{_cdir}/{logsig_cache_key}_train.pt'
                _cte = f'{_cdir}/{logsig_cache_key}_test.pt'
                _lock_path = f'{_cdir}/{logsig_cache_key}.lock'
                with open(_lock_path, 'w') as _lock_f:
                    fcntl.flock(_lock_f, fcntl.LOCK_EX)
                    if os.path.exists(_ctr) and os.path.exists(_cte):
                        print(f'[logsig cache hit] {_ctr}', flush=True)
                        data_tr = torch.load(_ctr, weights_only=True)
                        data_te = torch.load(_cte, weights_only=True)
                    else:
                        print(f'[logsig cache miss — computing] {_ctr}', flush=True)
                        data_tr = get_logsig(X_train_xt, **_logsig_kw)
                        data_te = get_logsig(X_test_xt,  **_logsig_kw)
                        torch.save(data_tr, f'{_ctr}.tmp')
                        torch.save(data_te, f'{_cte}.tmp')
                        os.rename(f'{_ctr}.tmp', _ctr)
                        os.rename(f'{_cte}.tmp', _cte)
                    fcntl.flock(_lock_f, fcntl.LOCK_UN)
            else:
                data_tr = get_logsig(X_train_xt, **_logsig_kw)
                data_te = get_logsig(X_test_xt,  **_logsig_kw)
            if logsig_normalize:
                _n_copies = len(logsig_multi_smooth_params) if logsig_multi_smooth_params else 1
                _nlvl_kw = dict(
                    depth=logsig_depth,
                    path_dim=X_train_xt.shape[-1] + 1,
                    num_copies=_n_copies,
                    global_time=logsig_global_time,
                )
                data_tr, _norm_stats = normalize_logsig_levels(data_tr, **_nlvl_kw)
                data_te, _ = normalize_logsig_levels(data_te, **_nlvl_kw, stats=_norm_stats)
            data_tr, data_te, mean, std = normalize(data_tr, data_te)
        else:
            raise ValueError(f"Unknown view '{view}'. Choose from: xt, dx, xf, logsig")

        results[key] = (data_tr.float(), data_te.float(), mean, std)

    return results


class Load_Dataset(Dataset):
    """N-view dataset — works for any number of views (2, 3, …).

    Args:
        X:          list of N tensors [num_samples, L, C], one per view.
        X_aug:      list of N tensors used as the augmentation source.
                    For pretrain pass the same tensors as X.
                    For finetune/test X_aug is ignored; X is used and the
                    augmentation function is applied in __getitem__.
        y:          label tensor [num_samples].
        mode:       'pretrain' | 'finetune' | 'test'.
        views:      ordered sequence of view names (same order as X),
                    e.g. ('xt', 'dx', 'xf') or ('xt', 'logsig').
        num_repeats: number of augmented copies generated per sample in
                    pretrain mode.
        aug_fns:    optional list of per-view augmentation callables
                    overriding _aug_fn_for_view.

    __getitem__ returns 2*N + 1 tensors:
        (view_0, …, view_{N-1}, aug_0, …, aug_{N-1}, y)
    """
    def __init__(self, X: list, X_aug: list, y: torch.Tensor,
                 mode: str, num_repeats: int = 1,
                 views=('xt', 'dx', 'xf'),
                 aug_fns: Optional[list] = None):
        super().__init__()
        self.mode      = mode
        self.views     = list(views)
        self.num_views = len(views)
        self.aug_fns   = aug_fns if aug_fns is not None else [_aug_fn_for_view(v) for v in views]

        if mode == 'pretrain':
            self.data     = [self._repeat(x, num_repeats) for x in X]
            self.data_aug = list(X_aug)
            self.y = y.long().unsqueeze(-1).repeat(1, num_repeats).reshape(-1)
        else:
            self.data     = [x.float() for x in X]
            self.data_aug = self.data   # augmentation applied per-sample in __getitem__
            self.y = y.long().reshape(-1)

    @staticmethod
    def _repeat(x: torch.Tensor, num_repeats: int) -> torch.Tensor:
        x = x.float().unsqueeze(-1).repeat(1, 1, 1, num_repeats)
        return x.permute(0, 3, 1, 2).reshape(-1, x.shape[1], x.shape[2])

    def __len__(self) -> int:
        return self.data[0].shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, ...]:
        orig = [self.data[i][idx]         for i in range(self.num_views)]
        aug  = [fn(self.data_aug[i][idx]) for i, fn in enumerate(self.aug_fns)]
        return (*orig, *aug, self.y[idx])

    # Augmentation helpers kept as static methods for backward compatibility
    @staticmethod
    def data_transform_td(sample: torch.Tensor, sigma: float = 0.1) -> torch.Tensor:
        return sample + torch.normal(mean=0., std=sigma, size=sample.shape, device=sample.device)

    @staticmethod
    def data_transform_fd(sample: torch.Tensor, pertub_ratio: float = 0.05) -> torch.Tensor:
        aug_1 = Load_Dataset.remove_frequency(sample, pertub_ratio)
        aug_2 = Load_Dataset.add_frequency(sample, pertub_ratio)
        return aug_1 + aug_2

    @staticmethod
    def remove_frequency(x: torch.Tensor, pertub_ratio: float = 0.0) -> torch.Tensor:
        mask = torch.rand(x.shape, device=x.device) > pertub_ratio
        return x * mask

    @staticmethod
    def add_frequency(x: torch.Tensor, pertub_ratio: float = 0.0) -> torch.Tensor:
        mask = torch.rand(x.shape, device=x.device) > (1 - pertub_ratio)
        max_amplitude = x.max()
        random_am = torch.rand(mask.shape, device=x.device) * (max_amplitude * 0.1)
        pertub_matrix = mask * random_am
        return x + pertub_matrix


def _aug_fn_for_view(view: str):
    """Return the augmentation callable for a given view name.

    - 'xf'     -> frequency perturbation
    - 'logsig' -> identity (no augmentation applied to the signature)
    - else     -> additive Gaussian noise
    """
    if view == 'xf':
        return Load_Dataset.data_transform_fd
    elif view == 'logsig':
        return lambda x: x
    else:
        return Load_Dataset.data_transform_td


def _make_logsig_noise_aug(
    noise_scale: float,
    logsig_train_data: torch.Tensor,
    depth: int,
    path_dim: int,
    has_global_time: bool = False,
    num_copies: int = 1,
):
    """Return a logsig augmentation function that adds per-level Gaussian noise.

    For each depth-k level, noise ~ N(0, noise_scale * std_k) where std_k is
    the std of that level's features computed over the full training set.
    Setting noise_scale=0.1 adds noise at one tenth of each level's training
    std, automatically tracking the order of magnitude of each level.

    Args:
        noise_scale:        fraction of each level's training std to use as
                            noise std (e.g. 0.1 = 10 %).
        logsig_train_data:  preprocessed training tensor [N, T, C] used to
                            compute per-level stds.
        depth:              log-signature truncation depth.
        path_dim:           path dimensionality including the prepended time
                            channel (i.e. num_input_features + 1).
        has_global_time:    if True, a global-time scalar is appended after the
                            signature features — it is left unperturbed.
        num_copies:         K for multi_smooth_params (K stacked signatures).
    """
    level_bounds = []
    prev = 0
    for k in range(1, depth + 1):
        end = logsigdim(path_dim, k)
        level_bounds.append((prev, end))
        prev = end
    C_sig = prev  # logsigdim(path_dim, depth)

    # Pre-compute fixed noise stds from training data (one scalar per level per copy)
    level_noise_stds = []
    for c in range(num_copies):
        base = c * C_sig
        for lvl_start, lvl_end in level_bounds:
            sl = slice(base + lvl_start, base + lvl_end)
            train_std = float(logsig_train_data[..., sl].std().clamp(min=1e-8))
            level_noise_stds.append(noise_scale * train_std)

    def aug(x: torch.Tensor) -> torch.Tensor:
        out = x.clone()
        idx = 0
        for c in range(num_copies):
            base = c * C_sig
            for lvl_start, lvl_end in level_bounds:
                sl = slice(base + lvl_start, base + lvl_end)
                out[..., sl] = x[..., sl] + level_noise_stds[idx] * torch.randn_like(x[..., sl])
                idx += 1
        return out

    return aug


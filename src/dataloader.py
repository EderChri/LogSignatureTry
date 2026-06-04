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
            data_tr, data_te, mean, std = normalize(data_tr, data_te)
        else:
            raise ValueError(f"Unknown view '{view}'. Choose from: xt, dx, xf, logsig")

        results[key] = (data_tr.float(), data_te.float(), mean, std)

    return results


class Load_Dataset(Dataset):
    def __init__(self, X: list, X_aug: list, y: torch.Tensor,
                 mode: str, num_repeats: int = 1,
                 views: tuple = ('xt', 'dx', 'xf')):
        super(Load_Dataset, self).__init__()

        self.mode = mode
        self.num_repeats = num_repeats
        self.views = views

        if self.mode == 'pretrain':
            self.setup_pretrain_data(X, X_aug, y)
        else:
            self.setup_finetune_data(X, y)

    def setup_pretrain_data(self, X: list, X_aug: list, y: torch.Tensor):
        self.xt, self.dx, self.xf = X
        self.xt, self.dx, self.xf = self.get_repeats(self.xt), self.get_repeats(self.dx), self.get_repeats(self.xf)
        self.xt_aug, self.dx_aug, self.xf_aug = X_aug
        self.y = y.long().unsqueeze(-1).repeat(1, self.num_repeats).reshape(-1)

    def setup_finetune_data(self, X: torch.Tensor, y: torch.Tensor):
        self.xt, self.dx, self.xf = X
        self.xt_aug, self.dx_aug, self.xf_aug = X
        self.y = y.long().reshape(-1)

    def get_repeats(self, X: torch.Tensor, num_repeats: int = 10):
        X = X.float().unsqueeze(-1).repeat(1, 1, 1, self.num_repeats)
        return X.permute(0, 3, 1, 2).reshape(-1, X.shape[1], X.shape[2])

    def __len__(self) -> int:
        return self.xt.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, ...]:
        aug1, aug2, aug3 = [_aug_fn_for_view(v) for v in self.views]
        if self.mode == 'pretrain':
            return (self.xt_aug[idx], self.dx_aug[idx], self.xf_aug[idx],
                    aug1(self.xt_aug[idx]), aug2(self.dx_aug[idx]), aug3(self.xf_aug[idx]),
                    self.y[idx])
        else:
            return (self.xt[idx], self.dx[idx], self.xf[idx],
                    aug1(self.xt[idx]), aug2(self.dx[idx]), aug3(self.xf[idx]),
                    self.y[idx])

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


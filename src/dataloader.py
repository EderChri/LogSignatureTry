import torch
import torch.fft as fft
import torchcde
from log_signatures_pytorch import log_signature, logsigdim
from torch.utils.data import Dataset
from typing import Tuple, Optional


def normalize(X_train: torch.Tensor, X_test: torch.Tensor, epsilon: float = 1e-8):
    # Compute mean and std along the N and L dimensions
    mean = X_train.mean(dim=(0, 1), keepdim=True)
    std = X_train.std(dim=(0, 1), keepdim=True)
    
    # Add epsilon to std to avoid division by zero
    std = std.clamp(min=epsilon)
    
    # Normalize train and test data
    X_train_norm = (X_train - mean) / std
    X_test_norm = (X_test - mean) / std
    return X_train_norm, X_test_norm, mean, std  # Return mean and std for potential inverse transform


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
    return torch.abs(fft.fft(X, dim=1))  # Use dim=1 for the sequence dimension


def get_logsig(X: torch.Tensor, depth: int) -> torch.Tensor:
    """Running log signature of the time-augmented path.

    Prepends a t in [0,1] coordinate so the path lives in R^(D+1), then at
    each step t returns the log signature of the sub-path [0, t].  The time
    channel ensures the logsig is non-trivial even for single-channel inputs.

    Args:
        X:     [N, L, D]
        depth: truncation depth

    Returns:
        [N, L, logsig_channels(D+1, depth)]
        Position 0 is a zero vector (log sig of the empty path);
        position t > 0 is the log sig of the time-augmented path up to step t.
    """
    N, L, D = X.shape
    t = torch.linspace(0, 1, L, dtype=X.dtype, device=X.device)
    t = t.view(1, L, 1).expand(N, -1, -1)
    X_time = torch.cat([t, X], dim=-1)                       # [N, L, D+1]
    logsig = log_signature(X_time, depth, stream=True)  # [N, L-1, C]
    pad = torch.zeros(N, 1, logsig.shape[-1], dtype=X.dtype, device=X.device)
    return torch.cat([pad, logsig], dim=1)                   # [N, L, C]


def get_view_num_features(view: str, num_feature: int, logsig_depth: int) -> int:
    """Input feature dimension produced by the given view transform."""
    if view in ('xt', 'dx', 'xf'):
        return num_feature
    elif view == 'logsig':
        return logsigdim(num_feature + 1, logsig_depth)
    else:
        raise ValueError(f"Unknown view '{view}'. Choose from: xt, dx, xf, logsig")


def preprocess_data(X_train, X_test, views=('xt', 'dx', 'xf'), logsig_depth=2, time_as_feature=False):
    """Preprocess training and test data for the requested views.

    Args:
        views:        tuple of three view names; first entry must be 'xt'.
        logsig_depth: truncation depth used when a view is 'logsig'.

    Returns:
        dict with keys 'v1', 'v2', 'v3', each a tuple
        (X_train, X_test, mean, std).
    """
    # Normalise time-domain data — used as input for all other transforms
    X_train_xt, X_test_xt, mean_xt, std_xt = normalize(X_train, X_test)

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
            data_tr = get_logsig(X_train_xt, logsig_depth)
            data_te = get_logsig(X_test_xt, logsig_depth)
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


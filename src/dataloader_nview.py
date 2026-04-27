"""dataloader_nview.py — N-view dataset for EncoderNView / ClassifierNView.

Drop-in replacement for Load_Dataset when using more or fewer than 3 views.

preprocess_data() from src/dataloader.py already handles arbitrary view lists,
so only the Dataset class needs a new implementation.

Usage
-----
    from src.dataloader import preprocess_data, get_view_num_features
    from src.dataloader_nview import Load_DatasetNView

    views   = ['xt', 'logsig']       # two views
    in_dims = [num_feature, logsig_dim]

    pre = preprocess_data(X_tr, X_te, views=views, ...)
    # pre has keys 'v1', 'v2' (one per view)

    ds = Load_DatasetNView(
        views_data    = [pre['v1'][0], pre['v2'][0]],   # train split
        views_aug_data= [pre['v1'][0], pre['v2'][0]],
        y             = y_train,
        mode          = 'pretrain',
        views         = views,
    )

    loader = DataLoader(ds, batch_size=64, ...)
    # Each batch: (*views_orig, *views_aug, y) — 2*N+1 tensors.
    # Unpack with: views_orig = batch[:N]; views_aug = batch[N:2*N]; y = batch[2*N]
"""

import torch
from torch.utils.data import Dataset
from .dataloader import _aug_fn_for_view


class Load_DatasetNView(Dataset):
    """N-view dataset — generalises Load_Dataset to any number of views.

    Args:
        views_data:      list of N tensors [num_samples, L, C], one per view.
        views_aug_data:  list of N tensors used as the augmentation source.
                         For pretrain pass the same tensors as views_data.
                         For finetune/test the argument is ignored (views_data
                         is used and the augmentation function is applied in
                         __getitem__).
        y:               label tensor [num_samples].
        mode:            'pretrain' | 'finetune' | 'test'.
        views:           ordered list of view names (same order as views_data),
                         e.g. ['xt', 'logsig'].  Used to select the augmentation
                         function per view.
        num_repeats:     number of augmented copies generated per sample in
                         pretrain mode (matches Load_Dataset behaviour).

    __getitem__ returns 2*N + 1 tensors:
        (view_0, view_1, ..., view_{N-1},
         aug_0,  aug_1,  ..., aug_{N-1},
         y)
    """
    def __init__(self, views_data: list, views_aug_data: list, y: torch.Tensor,
                 mode: str, views: list, num_repeats: int = 1):
        super().__init__()
        assert len(views_data) == len(views_aug_data) == len(views), (
            "views_data, views_aug_data, and views must all have the same length"
        )

        self.mode      = mode
        self.views     = list(views)
        self.num_views = len(views)
        self.aug_fns   = [_aug_fn_for_view(v) for v in views]

        if mode == 'pretrain':
            self.data     = [self._repeat(x, num_repeats) for x in views_data]
            self.data_aug = list(views_aug_data)
            self.y = y.long().unsqueeze(-1).repeat(1, num_repeats).reshape(-1)
        else:
            self.data     = [x.float() for x in views_data]
            self.data_aug = self.data   # augmentation applied per-sample in __getitem__
            self.y = y.long().reshape(-1)

    @staticmethod
    def _repeat(x: torch.Tensor, num_repeats: int) -> torch.Tensor:
        x = x.float().unsqueeze(-1).repeat(1, 1, 1, num_repeats)
        return x.permute(0, 3, 1, 2).reshape(-1, x.shape[1], x.shape[2])

    def __len__(self) -> int:
        return self.data[0].shape[0]

    def __getitem__(self, idx: int):
        orig = [self.data[i][idx]              for i in range(self.num_views)]
        aug  = [fn(self.data_aug[i][idx])      for i, fn in enumerate(self.aug_fns)]
        return (*orig, *aug, self.y[idx])

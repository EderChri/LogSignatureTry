"""model_nview.py — N-view generalisation of the MVCL encoder and classifier.

Supports any number of views ≥ 2.  Drop-in replacement for the 3-view
Encoder/Classifier in src/model.py for experiments with fewer or more views.

Key differences from src/model.py:
  - InteractionLayerNView: num_views is a constructor argument, not hardcoded 3.
  - EncoderNView: one encoder branch per view (nn.ModuleList), indexed not named.
    Returns (hiddens: list, projections: list) instead of a 6-tuple.
  - ClassifierNView: always uses all N views (loss_type='ALL' generalisation).
    Accepts a list of tensors rather than three separate positional arguments.
"""

import torch
import torch.nn as nn

from .model import PositionalEncoding, LogSigMLP, SelfAttention, _uses_mlp_for_view, _pool


# ---------------------------------------------------------------------------
# Internal branch: Linear projection + PE + TransformerEncoder
# ---------------------------------------------------------------------------

class _TransformerBranch(nn.Module):
    """Full transformer pipeline for one view: proj → PE → TransformerEncoder."""
    def __init__(self, in_dim, embedding_dim, hidden_dim, num_head, num_layers, dropout):
        super().__init__()
        self.proj = nn.Linear(in_dim, embedding_dim)
        self.pe   = PositionalEncoding(embedding_dim, dropout)
        self.enc  = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=embedding_dim, dim_feedforward=hidden_dim,
                                       nhead=num_head, dropout=dropout, batch_first=True),
            num_layers,
        )

    def forward(self, x):  # [N, L, in_dim] -> [N, L, embedding_dim]
        return self.enc(self.pe(self.proj(x)))


# ---------------------------------------------------------------------------
# N-view interaction layer
# ---------------------------------------------------------------------------

class InteractionLayerNView(nn.Module):
    """Cross-view multi-head attention for N views.

    Generalises InteractionLayer from src/model.py to variable N.
    Identical behaviour when num_views=3.
    """
    def __init__(self, hidden_size: int, num_heads: int, num_views: int):
        super().__init__()
        self.num_views = num_views
        self.mha  = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=num_heads,
                                           batch_first=True)
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, *hs, return_attn: bool = False):
        """
        hs:         num_views tensors, each [N, L, D]
        return_attn if True, also returns attention weights [N, L, V, V]
        Returns:    list of num_views tensors [N, L, D]
        """
        V = self.num_views
        N, L, D = hs[0].size()
        h = torch.stack(list(hs), dim=2).view(N * L, V, D)   # [N*L, V, D]
        attn_out, attn_w = self.mha(h, h, h)
        out = self.norm(h + attn_out).view(N, L, V, D)
        result = [out[:, :, i, :] for i in range(V)]
        if return_attn:
            return result, attn_w.view(N, L, V, V)
        return result


# ---------------------------------------------------------------------------
# N-view encoder
# ---------------------------------------------------------------------------

class EncoderNView(nn.Module):
    """N-view contrastive encoder.

    Generalises src.model.Encoder to any number of views ≥ 2.

    View 0 must be 'xt' and always uses a Transformer branch.
    Subsequent views use a Transformer or LogSigMLP, controlled by
    args.encoder_type and the view name.

    Args:
        args:    namespace with num_embedding, num_hidden, num_head,
                 num_layers, dropout, and optionally encoder_type.
        views:   ordered list of view names.  First element must be 'xt'.
                 e.g. ['xt', 'logsig'] or ['xt', 'dx', 'xf', 'logsig'].
        in_dims: input feature dimension for each view (same length as views).

    forward(*xs) returns:
        hiddens     — list of [N, L, D] tensors, one per view
        projections — list of [N, num_hidden] tensors, one per view
    """
    def __init__(self, args, views: list, in_dims: list):
        super().__init__()
        assert views[0] == 'xt',            "First view must be 'xt'"
        assert len(views) == len(in_dims),  "views and in_dims must have the same length"
        assert len(views) >= 2,             "Need at least 2 views"

        encoder_type   = getattr(args, 'encoder_type', 'transformer')
        self.views     = list(views)
        self.num_views = len(views)
        # view 0 (xt) never uses MLP even with encoder_type='mlp_logsig'
        self._use_last = [False] + [_uses_mlp_for_view(encoder_type, v) for v in views[1:]]

        branch_kw = dict(embedding_dim=args.num_embedding, hidden_dim=args.num_hidden,
                         num_head=args.num_head, num_layers=args.num_layers,
                         dropout=args.dropout)

        self.branches = nn.ModuleList([
            LogSigMLP(in_dims[i], args.num_embedding, args.num_hidden, args.dropout)
            if self._use_last[i] else
            _TransformerBranch(in_dims[i], **branch_kw)
            for i in range(self.num_views)
        ])

        self.interaction_layer = InteractionLayerNView(
            args.num_embedding, args.num_head, self.num_views
        )

        self.output_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(args.num_embedding * 2, args.num_hidden),
                nn.LayerNorm(args.num_hidden), nn.ReLU(), nn.Dropout(args.dropout),
                nn.Linear(args.num_hidden, args.num_hidden),
            )
            for _ in views
        ])

    def forward(self, *xs):
        assert len(xs) == self.num_views, (
            f"Expected {self.num_views} view tensors, got {len(xs)}"
        )
        hiddens    = [branch(torch.nan_to_num(x)) for branch, x in zip(self.branches, xs)]
        interacted = self.interaction_layer(*hiddens)

        projections = [
            self.output_layers[i](torch.cat([
                _pool(hiddens[i],    self._use_last[i]),
                _pool(interacted[i], self._use_last[i]),
            ], dim=-1))
            for i in range(self.num_views)
        ]
        return hiddens, projections


# ---------------------------------------------------------------------------
# N-view classifier
# ---------------------------------------------------------------------------

class ClassifierNView(nn.Module):
    """Classification head for an N-view encoder.

    Always concatenates representations from all N views (generalisation of
    loss_type='ALL' from the 3-view Classifier in src/model.py).

    Supports:
        feature='hidden'  — applies its own interaction + output layers to the
                            encoder's hidden states, then feeds to a linear head.
        feature='latent'  — applies self-attention over the encoder's projections
                            then feeds to a linear head.

    forward(xs) where xs is a list of N tensors:
        feature='hidden' → each tensor is [N_batch, L, D] (hidden states)
        feature='latent' → each tensor is [N_batch, num_hidden] (projections)
    """
    def __init__(self, args, views: list):
        super().__init__()
        self.args      = args
        self.views     = list(views)
        self.num_views = len(views)
        encoder_type   = getattr(args, 'encoder_type', 'transformer')
        self._use_last = [False] + [_uses_mlp_for_view(encoder_type, v) for v in views[1:]]

        if args.feature == 'hidden':
            self.interaction_layer = InteractionLayerNView(
                args.num_embedding, args.num_head, self.num_views
            )
            self.output_layers = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(args.num_embedding * 2, args.num_hidden),
                    nn.LayerNorm(args.num_hidden), nn.ReLU(), nn.Dropout(args.dropout),
                    nn.Linear(args.num_hidden, args.num_hidden),
                )
                for _ in views
            ])

        elif args.feature == 'latent':
            # SelfAttention from src.model works for any sequence length V
            self.self_attention = SelfAttention(args.num_hidden)

        self.fc = nn.Linear(self.num_views * args.num_hidden, args.num_target)
        self.fc.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.01)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, xs: list):
        xs = [torch.nan_to_num(x) for x in xs]

        if self.args.feature == 'latent':
            stacked  = torch.stack(xs, dim=1)                      # [N, V, H]
            attended = self.self_attention(stacked)[0] + stacked    # residual
            zs = [attended[:, i, :] for i in range(self.num_views)]

        else:  # 'hidden'
            interacted = self.interaction_layer(*xs)
            zs = [
                self.output_layers[i](torch.cat([
                    _pool(xs[i],         self._use_last[i]),
                    _pool(interacted[i], self._use_last[i]),
                ], dim=-1))
                for i in range(self.num_views)
            ]

        return self.fc(torch.cat(zs, dim=-1))

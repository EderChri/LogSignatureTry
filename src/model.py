import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    def __init__(self, hidden_dim, dropout=0.1, max_len=1024):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, hidden_dim)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, hidden_dim, 2) * 
                             (-math.log(10000.0) / hidden_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # Shape: [1, max_len, hidden_dim]
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class InteractionLayer(nn.Module):
    def __init__(self, hidden_size, num_heads):
        super(InteractionLayer, self).__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            batch_first=True
        )
        self.norm = nn.LayerNorm(hidden_size)
        
    def forward(self, ht, hd, hf, return_attn=False):
        N, L, D = ht.size()
        h = torch.stack([ht, hd, hf], dim=2)  # [N, L, 3, D]
        h = h.view(N * L, 3, D)               # [N*L, 3, D] — attention across views at each timestep

        attn_output, attn_weights = self.multihead_attn(h, h, h)  # attn_weights: [N*L, 3, 3]
        output = self.norm(h + attn_output)
        output = output.view(N, L, 3, D)

        ht_i, hd_i, hf_i = output[:, :, 0, :], output[:, :, 1, :], output[:, :, 2, :]
        if return_attn:
            # reshape to [N, L, 3, 3]: (query_view, key_view) attention at each timestep
            return ht_i, hd_i, hf_i, attn_weights.view(N, L, 3, 3)
        return ht_i, hd_i, hf_i

        
def _uses_mlp_for_view(encoder_type: str, view: str) -> bool:
    """Return True if this view should use LogSigMLP instead of a transformer."""
    return encoder_type == 'mlp_logsig' and view == 'logsig'


def _pool(h: torch.Tensor, use_last: bool) -> torch.Tensor:
    return h[:, -1, :] if use_last else h.mean(dim=1)


class LogSigMLP(nn.Module):
    """Per-timestep MLP for log signature views.

    Replaces the input_layer + positional_encoding + transformer_encoder pipeline.
    No positional encoding is applied since time is already encoded in the log signature.
    Stored as input_layer_d/f so that freeze mode (which checks 'input_layer' in param
    name) correctly unfreezes the MLP along with the classifier.
    """
    def __init__(self, in_dim, embedding_dim, hidden_dim, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embedding_dim),
        )

    def forward(self, x):  # [N, L, C] -> [N, L, embedding_dim]
        return self.net(x)


class Encoder(nn.Module):
    """Multi-view encoder.

    View 1 (temporal) always uses a Transformer.
    Views 2 & 3 use a Transformer by default (encoder_type='transformer') or a
    per-timestep LogSigMLP when encoder_type='mlp_logsig' AND the view is 'logsig'.
    """
    def __init__(self, args):
        super().__init__()
        encoder_type = getattr(args, 'encoder_type', 'transformer')
        self.view2 = args.view2
        self.view3 = args.view3
        self._v2_mlp = _uses_mlp_for_view(encoder_type, args.view2)
        self._v3_mlp = _uses_mlp_for_view(encoder_type, args.view3)

        self.positional_encoding = PositionalEncoding(args.num_embedding, args.dropout)

        # View 1 — always transformer
        self.input_layer_t = nn.Linear(args.num_feature, args.num_embedding)
        self.transformer_encoder_t = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=args.num_embedding, dim_feedforward=args.num_hidden,
                                       nhead=args.num_head, dropout=args.dropout, batch_first=True),
            args.num_layers)

        # View 2
        in_dim_d = getattr(args, 'num_feature_v2', args.num_feature)
        if self._v2_mlp:
            self.input_layer_d = LogSigMLP(in_dim_d, args.num_embedding, args.num_hidden, args.dropout)
        else:
            self.input_layer_d = nn.Linear(in_dim_d, args.num_embedding)
            self.transformer_encoder_d = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(d_model=args.num_embedding, dim_feedforward=args.num_hidden,
                                           nhead=args.num_head, dropout=args.dropout, batch_first=True),
                args.num_layers)

        # View 3
        in_dim_f = getattr(args, 'num_feature_v3', args.num_feature)
        if self._v3_mlp:
            self.input_layer_f = LogSigMLP(in_dim_f, args.num_embedding, args.num_hidden, args.dropout)
        else:
            self.input_layer_f = nn.Linear(in_dim_f, args.num_embedding)
            self.transformer_encoder_f = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(d_model=args.num_embedding, dim_feedforward=args.num_hidden,
                                           nhead=args.num_head, dropout=args.dropout, batch_first=True),
                args.num_layers)

        self.interaction_layer = InteractionLayer(args.num_embedding, args.num_head)
        self.output_layer_t = nn.Sequential(
            nn.Linear(args.num_embedding * 2, args.num_hidden),
            nn.LayerNorm(args.num_hidden), nn.ReLU(), nn.Dropout(args.dropout),
            nn.Linear(args.num_hidden, args.num_hidden))
        self.output_layer_d = nn.Sequential(
            nn.Linear(args.num_embedding * 2, args.num_hidden),
            nn.LayerNorm(args.num_hidden), nn.ReLU(), nn.Dropout(args.dropout),
            nn.Linear(args.num_hidden, args.num_hidden))
        self.output_layer_f = nn.Sequential(
            nn.Linear(args.num_embedding * 2, args.num_hidden),
            nn.LayerNorm(args.num_hidden), nn.ReLU(), nn.Dropout(args.dropout),
            nn.Linear(args.num_hidden, args.num_hidden))

    def forward(self, xt, dx, xf):
        xt, dx, xf = torch.nan_to_num(xt), torch.nan_to_num(dx), torch.nan_to_num(xf)

        ht = self.positional_encoding(self.input_layer_t(xt))
        ht = self.transformer_encoder_t(ht)

        if self._v2_mlp:
            hd = self.input_layer_d(dx)
        else:
            hd = self.positional_encoding(self.input_layer_d(dx))
            hd = self.transformer_encoder_d(hd)

        if self._v3_mlp:
            hf = self.input_layer_f(xf)
        else:
            hf = self.positional_encoding(self.input_layer_f(xf))
            hf = self.transformer_encoder_f(hf)

        ht_i, hd_i, hf_i = self.interaction_layer(ht, hd, hf)

        zt = self.output_layer_t(torch.cat([ht.mean(dim=1), ht_i.mean(dim=1)], dim=-1))
        # For logsig, position -1 is the global log signature; mean pooling would
        # dilute it with near-empty early prefix sigs.
        zd = self.output_layer_d(torch.cat([_pool(hd, self._v2_mlp), _pool(hd_i, self._v2_mlp)], dim=-1))
        zf = self.output_layer_f(torch.cat([_pool(hf, self._v3_mlp), _pool(hf_i, self._v3_mlp)], dim=-1))

        return ht, hd, hf, zt, zd, zf


# Backward-compatibility alias
EncoderLogsigMLP = Encoder


class SelfAttention(nn.Module):
    def __init__(self, hidden_dim):
        super(SelfAttention, self).__init__()
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(self, x):
        q = self.query(x)  # (batch_size, 3, hidden_dim)
        k = self.key(x)    # (batch_size, 3, hidden_dim)
        v = self.value(x)  # (batch_size, 3, hidden_dim)
        
        scores = torch.matmul(q, k.transpose(-2, -1)) / (x.size(-1) ** 0.5)  # (batch_size, 3, 3)
        attention_weights = F.softmax(scores, dim=-1)  # (batch_size, 3, 3)
        output = torch.matmul(attention_weights, v)  # (batch_size, 3, hidden_dim)
        return output, attention_weights
        

class Classifier(nn.Module):
    def __init__(self, args):
        super(Classifier, self).__init__()
        self.args = args
        if self.args.feature == 'latent':
            if args.loss_type == 'ALL':
                self.self_attention = SelfAttention(args.num_hidden)

        elif self.args.feature == 'hidden':
            ## interaction
            self.interaction_layer = InteractionLayer(args.num_embedding, args.num_head)
            
            ## output
            self.output_layer_t = nn.Sequential(
                nn.Linear(args.num_embedding*2, args.num_hidden),
                nn.LayerNorm(args.num_hidden), nn.ReLU(), nn.Dropout(args.dropout),
                nn.Linear(args.num_hidden, args.num_hidden)
            )
            self.output_layer_d = nn.Sequential(
                nn.Linear(args.num_embedding*2, args.num_hidden),
                nn.LayerNorm(args.num_hidden), nn.ReLU(), nn.Dropout(args.dropout),
                nn.Linear(args.num_hidden, args.num_hidden)
            )
            self.output_layer_f = nn.Sequential(
                nn.Linear(args.num_embedding*2, args.num_hidden),
                nn.LayerNorm(args.num_hidden), nn.ReLU(), nn.Dropout(args.dropout),
                nn.Linear(args.num_hidden, args.num_hidden)
            )

        self.fc = nn.Linear(len(args.loss_type)*args.num_hidden, args.num_target)
        
        self.fc.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.01)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
    
    def forward(self, xt, dx, xf):
        xt, dx, xf = torch.nan_to_num(xt), torch.nan_to_num(dx), torch.nan_to_num(xf)

        if self.args.feature == 'latent':
            zt, zd, zf = xt, dx, xf

            if self.args.loss_type == 'ALL':
                stacked_emb = torch.stack([zt, zd, zf], dim=1) # [batch_size, 3, hidden_dim]
                emb = self.self_attention(stacked_emb)[0] + stacked_emb # [batch_size, 3, hidden_dim]
                zt, zd, zf = emb[:, 0, :], emb[:, 1, :], emb[:, 2, :]
        
        elif self.args.feature == 'hidden':
            ht, hd, hf = xt, dx, xf
    
            # interaction
            if self.args.loss_type == 'ALL':
                ht_i, hd_i, hf_i = self.interaction_layer(ht, hd, hf)
            else:
                ht_i, hd_i, hf_i = ht, hd, hf
            
            enc_type = getattr(self.args, 'encoder_type', 'transformer')
            v2_last = _uses_mlp_for_view(enc_type, getattr(self.args, 'view2', ''))
            v3_last = _uses_mlp_for_view(enc_type, getattr(self.args, 'view3', ''))
            zt = self.output_layer_t(torch.cat([ht.mean(dim=1), ht_i.mean(dim=1)], dim=-1))
            zd = self.output_layer_d(torch.cat([_pool(hd, v2_last), _pool(hd_i, v2_last)], dim=-1))
            zf = self.output_layer_f(torch.cat([_pool(hf, v3_last), _pool(hf_i, v3_last)], dim=-1))
        
        if self.args.loss_type == 'ALL':
            emb = torch.cat([zt, zd, zf], dim=-1)
        else:
            emb_list = []
            # append embeddings based on the selected loss type
            if ('T' in self.args.loss_type):
                emb_list.append(zt)
            if ('D' in self.args.loss_type):
                emb_list.append(zd)
            if ('F' in self.args.loss_type):
                emb_list.append(zf)
            
            emb = torch.cat(emb_list, dim=-1)
            
        emb = emb.reshape(emb.shape[0], -1)
        return self.fc(emb)

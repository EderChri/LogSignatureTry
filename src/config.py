import argparse


def get_args_parser():
    parser = argparse.ArgumentParser()

    # Setup
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--data_name', default='_DA_HARTH_256_00', type=str)
    parser.add_argument('--num_feature', default=6, type=int)
    parser.add_argument('--num_target', default=4, type=int)

    # Data parameters
    parser.add_argument('--full_training', action='store_true', help='Enable full training mode (default: False)')
    parser.add_argument('--batch_size_pretrain', default=128, type=int)
    parser.add_argument('--batch_size_finetune', default=16, type=int)

    # Model parameters
    parser.add_argument('--encoder_type', default='transformer', type=str,
                        choices=['transformer', 'mlp_logsig'],
                        help='Per-view encoder: transformer (default) or mlp_logsig')
    parser.add_argument('--num_embedding', default=64, type=int)
    parser.add_argument('--num_hidden', default=128, type=int)
    parser.add_argument('--num_head', default=4, type=int)
    parser.add_argument('--num_layers', default=3, type=int)
    parser.add_argument('--dropout', default=0.2, type=float)
    parser.add_argument('--no_interaction_residual', action='store_true',
                        help='Remove the residual connection in InteractionLayer, forcing '
                             'all information to flow through cross-view attention')
    parser.add_argument('--interaction_type', default='attention', type=str,
                        choices=['attention', 'view_embed', 'bilinear', 'cross_time'],
                        help='InteractionLayer variant: '
                             'attention (default, cross-feature MHA at each timestep), '
                             'view_embed (learnable per-view offset added before MHA), '
                             'bilinear (asymmetric per-pair W_ij bilinear score), '
                             'cross_time (original-paper: temporal self-attn per view, no cross-view)')
    parser.add_argument('--feature', default='hidden', type=str)

    # Training parameters
    parser.add_argument('--random_attn_init', action='store_true',
                        help='Reinitialise MHA projection weights with N(0,1) to break '
                             'uniform attention at t=0 (diagnostic use)')
    parser.add_argument('--epochs_pretrain', default=2, type=int)
    parser.add_argument('--save_every', default=0, type=int,
                        help='Save a named epoch checkpoint every N pretrain epochs '
                             '(0 = disabled). Checkpoints go to '
                             'model_pretrain/{dataset}/epoch_ckpts/{tag}_ep{e}.pth')
    parser.add_argument('--epochs_finetune', default=100, type=int)
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--weight_decay', default=1e-5, type=float)
    parser.add_argument('--loss_type', default='ALL', type=str)
    parser.add_argument('--temperature', default=0.07, type=float)
    parser.add_argument('--lam', default=0.0, type=float)
    parser.add_argument('--partial', default=1.0, type=float)
    parser.add_argument('--cross_view_logsig', action='store_true',
                        help='Replace logsig intra-view loss with cross-view NT-Xent '
                             '(xt↔logsig and dx↔logsig). Adds clean pre-interaction '
                             'projection heads; logsig identity-aug term is dropped.')
    parser.add_argument('--lam_cross', default=1.0, type=float,
                        help='Weight for cross-view NT-Xent terms when --cross_view_logsig is set')

    # Cross-dataset transfer: pretrain on one dataset, finetune on another
    parser.add_argument('--pretrain_data_name', default=None, type=str,
                        help='data_name of the pretrained model; defaults to --data_name')

    # Which finetune modes to run
    parser.add_argument('--run_modes', default='finetune,freeze,baseline', type=str,
                        help='Comma-separated subset of finetune modes to run: '
                             'finetune, freeze, baseline (default: all three)')

    # View configuration (view1 is always 'xt')
    parser.add_argument('--view2', default='dx', type=str,
                        help="Second view: 'dx', 'xf', or 'logsig'")
    parser.add_argument('--view3', default='xf', type=str,
                        help="Third view: 'dx', 'xf', or 'logsig'")

    # Log signature options
    logsig = parser.add_argument_group('Log signature')
    logsig.add_argument('--logsig_depth', default=2, type=int,
                        help='Truncation depth for log signature')
    logsig.add_argument('--logsig_mode', default='stream', type=str,
                        choices=['stream', 'window', 'window_smooth'],
                        help='How to compute the log signature: '
                             'stream = running log-sig of [0,t] (default); '
                             'window = log-sig over a sliding window; '
                             'window_smooth = sliding window with smoothing before signature')
    logsig.add_argument('--logsig_window_size', default=32, type=int,
                        help='Window length for window/window_smooth modes')
    logsig.add_argument('--logsig_smoothing', default='tukey', type=str,
                        choices=['tukey', 'ema'],
                        help='Smoothing applied to each window before log-sig (window_smooth mode): '
                             'tukey = tapered-cosine window weighting; ema = exponential moving average')
    logsig.add_argument('--logsig_smooth_param', default=0.5, type=float,
                        help='Tukey alpha tapering ratio (0=rect, 1=Hann) or EMA decay alpha')
    logsig.add_argument('--logsig_global_time', action='store_true',
                        help='Append global t∈[0,1] as an extra channel to the logsig output '
                             '(gives windowed+MLP branch a sense of absolute position)')
    logsig.add_argument('--logsig_stride', default=1, type=int,
                        help='Compute windowed log-sig every stride steps (stride=1 = every step). '
                             'stride>1 activates InteractionLayerStridedLogsig in place of the '
                             'standard timestep-wise interaction layer. Only meaningful for '
                             'window/window_smooth modes.')
    logsig.add_argument('--logsig_pool', default='auto', type=str,
                        choices=['auto', 'last', 'mean'],
                        help='Pooling over the logsig time dimension after the encoder branch. '
                             'auto (default): last-token for mlp_logsig+stream, mean otherwise. '
                             'last: always last-token for logsig views (override for ablation). '
                             'mean: always mean-pool for logsig views (override for ablation).')
    logsig.add_argument('--logsig_multi_smooth_params', default=None, type=str,
                        help='Comma-separated Tukey alpha values for multi-param window_smooth. '
                             'E.g. "0.25,0.5,0.75" computes one log-signature per alpha and '
                             'concatenates them along the feature dim (output = K × C_sig). '
                             'Only effective in window_smooth mode. Overrides --logsig_smooth_param.')
    logsig.add_argument('--logsig_level_contrast', action='store_true',
                        help='Split log-signature into level-1 and level-2+ components and treat '
                             'each as a separate view. Adds a cross-level NTXentLoss between the '
                             'two level projections. Only effective with --view2 logsig in the '
                             'nview scripts. Requires --logsig_depth >= 2.')
    logsig.add_argument('--logsig_noise_scale', default=0.0, type=float,
                        help='Add per-level Gaussian noise to log-signature views as augmentation. '
                             'Noise std for depth-k features = noise_scale * std_k, where std_k '
                             'is the std of that level computed over the training set. '
                             'E.g. 0.1 adds noise at 10%% of each level\'s training std, '
                             'automatically tracking the order of magnitude of each level. '
                             '0.0 = disabled (default). Ignored when --cross_view_logsig is set.')
    logsig.add_argument('--logsig_normalize', action='store_true',
                        help='Normalize log-signature features per truncation level (z-score with '
                             'one shared mean/std per level, computed over all samples and timesteps). '
                             'Disabled by default for legacy compatibility.')
    logsig.add_argument('--logsig_lag', default=0, type=int,
                        help='Temporal lag (in timesteps) for the logsig contrastive positive. '
                             'When > 0 and logsig_mode is window/window_smooth, the positive pair '
                             'for the logsig view is the signature of the window l steps earlier: '
                             'logsig_lag[t] = logsig[t-l], zero-padded for t < l. '
                             'Replaces the identity/noise augmentation for the logsig view. '
                             'Only meaningful in window/window_smooth mode. 0 = disabled (default).')

    # Dimensionality reduction
    parser.add_argument('--pca_components', default=None, type=int,
                        help='If set, reduce input channels to this many PCA components before '
                             'computing all views. Fit on training data only. Useful for '
                             'high-dimensional datasets (e.g. Opportunity 113ch → 32, Skoda 60ch → 4).')

    return parser


def parse_args(args=None):
    parser = get_args_parser()
    parsed_args = parser.parse_args(args=args)
    return parsed_args
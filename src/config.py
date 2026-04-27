import argparse


def get_args_parser():
    parser = argparse.ArgumentParser()

    # Setup
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--data_name', default='BasicMotions_256_00', type=str)
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
    parser.add_argument('--feature', default='hidden', type=str)

    # Training parameters
    parser.add_argument('--epochs_pretrain', default=200, type=int)
    parser.add_argument('--epochs_finetune', default=100, type=int)
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--weight_decay', default=1e-5, type=float)
    parser.add_argument('--loss_type', default='ALL', type=str)
    parser.add_argument('--temperature', default=0.07, type=float)
    parser.add_argument('--lam', default=0.0, type=float)
    parser.add_argument('--partial', default=1.0, type=float)

    # Cross-dataset transfer: pretrain on one dataset, finetune on another
    parser.add_argument('--pretrain_data_name', default=None, type=str,
                        help='data_name of the pretrained model; defaults to --data_name')

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

    return parser


def parse_args(args=None):
    parser = get_args_parser()
    parsed_args = parser.parse_args(args=args)
    return parsed_args
import os
import sys
import math
import random
import argparse
import pickle 
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

import torch
from torch import nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import torchcde
from pytorch_metric_learning import losses

from src.config import *
from src.dataloader import *
from src.model import *
from src.trainer import *
from src.evaluation import *
from src.utils import *

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")


def _build_new_num_features(args):
    """Build the new_num_features dict for load_encoder.

    For mlp_logsig encoder, logsig views use LogSigMLP whose state-dict keys
    differ from a plain Linear, so dimension checking is skipped for those slots.
    """
    d = {'input_layer_t': args.num_feature}
    if not (args.encoder_type == 'mlp_logsig' and args.view2 == 'logsig'):
        d['input_layer_d'] = args.num_feature_v2
    if not (args.encoder_type == 'mlp_logsig' and args.view3 == 'logsig'):
        d['input_layer_f'] = args.num_feature_v3
    return d


def write_final_metric_row(summary_path, run_name, final_test_score, epochs_trained):
    if not os.path.exists(summary_path):
        with open(summary_path, 'w') as f:
            f.write('run_name\tfinal_test_score\tepochs_trained\n')

    with open(summary_path, 'a') as f:
        f.write(f'{run_name}\t{final_test_score:.6f}\t{epochs_trained}\n')


def seed_everything(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    
args = parse_args()
seed_everything(args.seed)

_enc_suffix = f'_{args.encoder_type}' if args.encoder_type != 'transformer' else ''

def _logsig_suffix(args) -> str:
    mode = getattr(args, 'logsig_mode', 'stream')
    if mode == 'stream':
        return ''
    wsiz = getattr(args, 'logsig_window_size', 32)
    if mode == 'window':
        return f'_win{wsiz}'
    smoothing = getattr(args, 'logsig_smoothing', 'tukey')
    return f'_{smoothing}{wsiz}'

_lsig_suffix = _logsig_suffix(args)

# Resolve pretrained model source dataset (defaults to the finetune dataset)
if args.pretrain_data_name is None:
    args.pretrain_data_name = args.data_name

#
if str(args.loss_type) not in ['ALL', 'TDF'] and float(args.lam) == 0.0:
    print("Error: check loss_type and lam")
    sys.exit(1)

#
args.context_len = int(args.data_name.split('_')[3])
args.horizon_len = int(args.data_name.split('_')[4])

#
if str(args.loss_type) not in ['ALL', 'TDF'] and args.horizon_len not in [0, 16]:
    print("Error: check loss_type and horizon")
    sys.exit(1)

## Load data
with open(f'preprocessed_data/{args.data_name}.pkl', 'rb') as f:
    X_train_intp, X_train_shirink, X_train_forecast, y_train, X_val_intp, X_val_shirink, X_val_forecast, y_val, X_test_intp, X_test_shirink, X_test_forecast, y_test = pickle.load(f)

##
X_train_intp = torch.tensor(X_train_intp).transpose(1,2)
y_train = torch.tensor(y_train)
X_val_intp = torch.tensor(X_val_intp).transpose(1,2)
y_val = torch.tensor(y_val)
X_test_intp = torch.tensor(X_test_intp).transpose(1,2)
y_test = torch.tensor(y_test)

##
views = ('xt', args.view2, args.view3)
_logsig_kw = dict(
    logsig_depth=args.logsig_depth,
    logsig_mode=getattr(args, 'logsig_mode', 'stream'),
    logsig_window_size=getattr(args, 'logsig_window_size', 32),
    logsig_smoothing=getattr(args, 'logsig_smoothing', 'tukey'),
    logsig_smooth_param=getattr(args, 'logsig_smooth_param', 0.5),
)

preprocessed_data = preprocess_data(X_train_intp, X_val_intp, views=views, **_logsig_kw)
X_train_intp_v1, X_val_intp_v1, _, _ = preprocessed_data['v1']
X_train_intp_v2, X_val_intp_v2, _, _ = preprocessed_data['v2']
X_train_intp_v3, X_val_intp_v3, _, _ = preprocessed_data['v3']

preprocessed_data = preprocess_data(X_train_intp, X_test_intp, views=views, **_logsig_kw)
X_train_intp_v1, X_test_intp_v1, _, _ = preprocessed_data['v1']
X_train_intp_v2, X_test_intp_v2, _, _ = preprocessed_data['v2']
X_train_intp_v3, X_test_intp_v3, _, _ = preprocessed_data['v3']

X_train = [X_train_intp_v1, X_train_intp_v2, X_train_intp_v3]
X_valid = [X_val_intp_v1, X_val_intp_v2, X_val_intp_v3]
X_test = [X_test_intp_v1, X_test_intp_v2, X_test_intp_v3]

train_dataset = Load_Dataset(X_train, X_train, y_train, 'finetune', views=views)
valid_dataset = Load_Dataset(X_valid, X_valid, y_val, 'test', views=views)
test_dataset = Load_Dataset(X_test, X_test, y_test, 'test', views=views)
train_loader = DataLoader(train_dataset, batch_size=args.batch_size_finetune, shuffle=True, drop_last=False, num_workers=4, pin_memory=True, persistent_workers=True)
valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size_finetune, shuffle=False, drop_last=False, num_workers=4, pin_memory=True, persistent_workers=True)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size_finetune, shuffle=False, drop_last=False, num_workers=4, pin_memory=True, persistent_workers=True)

##
os.makedirs(f'out_finetune', exist_ok=True)
os.makedirs(f'out_finetune/{args.data_name}', exist_ok=True)
summary_file = f'out_finetune/{args.data_name}/final_test_metric_summary.tsv'

# Dimension reduction with PCA
if args.num_feature > 64:
    args.num_feature = 64

args.num_feature_v2 = get_view_num_features(args.view2, args.num_feature, args.logsig_depth)
args.num_feature_v3 = get_view_num_features(args.view3, args.num_feature, args.logsig_depth)

pretrain_tag = f'{args.pretrain_data_name}_v2{args.view2}_v3{args.view3}_ep{args.epochs_pretrain}_{args.seed}{_enc_suffix}{_lsig_suffix}'
best_model_path = f'model_pretrain/{args.pretrain_data_name}/{pretrain_tag}.pth'

##
if len(args.loss_type) == 3:
    K = 1 # 5
else:
    K = 1

##
monitoring_metric = 'accuracy'

print(args)
for k in range(K):
    ## Run -- finetune
    if not os.path.exists(best_model_path):
        print(f"Pretrained checkpoint not found: {best_model_path}. Skipping this run.")
        continue

    if torch.cuda.device_count() > 1:
        encoder = Encoder(args)
        encoder = load_encoder(encoder, best_model_path, _build_new_num_features(args))
        encoder = nn.DataParallel(encoder).to(device)
        clf = Classifier(args)
        clf = nn.DataParallel(clf).to(device)
    else:
        encoder = Encoder(args)
        encoder = load_encoder(encoder, best_model_path, _build_new_num_features(args))
        encoder = encoder.to(device)
        clf = Classifier(args).to(device)
    
    for param in encoder.parameters():
        param.requires_grad = True
    
    encoder_optimizer = torch.optim.Adam(encoder.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    clf_optimizer = torch.optim.Adam(clf.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    encoder_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(encoder_optimizer, mode='max', factor=0.5, patience=10)
    clf_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(clf_optimizer, mode='max', factor=0.5, patience=10)
    
    loss_list = []
    metric_list = []
    best_valid_mm = 0
    best_valid_loss = float('inf')
    output_file = f'out_finetune/{args.data_name}/{args.data_name}_pt-{pretrain_tag}_{args.feature}_{args.loss_type}_{args.lam}_{k}_finetune'
    
    # Early stopping parameters
    patience = 20
    early_stop_counter = 0
    
    # Check if output already exists
    if os.path.exists(output_file):
        print(f"Output file {output_file} already exists. Skipping this run.")
    else:
        final_test_mm = None
        epochs_trained = 0
        # Run experiment
        for epoch in range(1, args.epochs_finetune + 1):
            train_loss, train_loss_c = train(args, encoder, clf, encoder_optimizer, clf_optimizer, train_loader, mode='finetune', device=device)
            valid_loss, valid_loss_c = test(args, encoder, clf, valid_loader, mode='valid', device=device)
            test_loss, test_loss_c = test(args, encoder, clf, test_loader, mode='test', device=device)
            
            print(f'Epoch {epoch}: Train Loss = {train_loss:.4f}, Validation Loss = {valid_loss:.4f}, Test Loss = {test_loss:.4f}')
            print(f'Epoch {epoch}: Train Loss_c = {train_loss_c:.4f}, Validation Loss_c = {valid_loss_c:.4f}, Test Loss_c = {test_loss_c:.4f}')
            loss_list.append([train_loss, valid_loss, test_loss, train_loss_c, valid_loss_c, test_loss_c])
            
            # Compute metrics
            train_metric = get_clf_metrics(args, encoder, clf, train_loader, device)
            valid_metric = get_clf_metrics(args, encoder, clf, valid_loader, device)
            test_metric = get_clf_metrics(args, encoder, clf, test_loader, device)
            
            train_mm = train_metric[monitoring_metric]
            valid_mm = valid_metric[monitoring_metric]
            test_mm = test_metric[monitoring_metric]
            
            print(f'Epoch {epoch}: Train Metric = {train_mm:.4f}, Validation Metric = {valid_mm:.4f}, Test Metric = {test_mm:.4f}')
            metric_list.append([train_metric, valid_metric, test_metric])
            final_test_mm = test_mm
            epochs_trained = epoch
            
            # Learning rate scheduling
            encoder_scheduler.step(valid_mm)
            clf_scheduler.step(valid_mm)
            
            # if valid_loss_c < best_valid_loss:
            #     best_valid_loss = valid_loss_c
            if valid_mm > best_valid_mm:
                best_valid_mm = valid_mm
                early_stop_counter = 0
            else:
                early_stop_counter += 1
            
            # Check for early stopping
            if early_stop_counter >= patience:
                print(f'Early stopping triggered at epoch {epoch}')
                break
        
        # Save final results
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'wb') as f:
            pickle.dump([args, loss_list, metric_list], f)

        if final_test_mm is not None:
            run_name = os.path.basename(output_file)
            write_final_metric_row(summary_file, run_name, final_test_mm, epochs_trained)
            print(f'Final summary written: {run_name}, score={final_test_mm:.4f}, epochs={epochs_trained}')
        

    ## Run -- freeze
    if torch.cuda.device_count() > 1:
        encoder = Encoder(args)
        encoder = load_encoder(encoder, best_model_path, _build_new_num_features(args))
        encoder = nn.DataParallel(encoder).to(device)
        clf = Classifier(args)
        clf = nn.DataParallel(clf).to(device)
    else:
        encoder = Encoder(args)
        encoder = load_encoder(encoder, best_model_path, _build_new_num_features(args))
        encoder = encoder.to(device)
        clf = Classifier(args).to(device)
    
    for name, param in encoder.named_parameters():
        if 'input_layer' in name:
            param.requires_grad = True
        else:
            param.requires_grad = False
    
    encoder_optimizer = torch.optim.Adam(encoder.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    clf_optimizer = torch.optim.Adam(clf.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    encoder_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(encoder_optimizer, mode='max', factor=0.5, patience=10)
    clf_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(clf_optimizer, mode='max', factor=0.5, patience=10)
    
    loss_list = []
    metric_list = []
    best_valid_mm = 0
    best_valid_loss = float('inf')
    output_file = f'out_finetune/{args.data_name}/{args.data_name}_pt-{pretrain_tag}_{args.feature}_{args.loss_type}_{args.lam}_{k}_freeze'
    
    # Early stopping parameters
    patience = 20
    early_stop_counter = 0
    
    # Check if output already exists
    if os.path.exists(output_file):
        print(f"Output file {output_file} already exists. Skipping this run.")
    else:
        final_test_mm = None
        epochs_trained = 0
        # Run experiment
        for epoch in range(1, args.epochs_finetune + 1):
            train_loss, train_loss_c = train(args, encoder, clf, encoder_optimizer, clf_optimizer, train_loader, mode='freeze', device=device)
            valid_loss, valid_loss_c = test(args, encoder, clf, valid_loader, mode='valid', device=device)
            test_loss, test_loss_c = test(args, encoder, clf, test_loader, mode='test', device=device)
            
            print(f'Epoch {epoch}: Train Loss = {train_loss:.4f}, Validation Loss = {valid_loss:.4f}, Test Loss = {test_loss:.4f}')
            print(f'Epoch {epoch}: Train Loss_c = {train_loss_c:.4f}, Validation Loss_c = {valid_loss_c:.4f}, Test Loss_c = {test_loss_c:.4f}')
            loss_list.append([train_loss, valid_loss, test_loss, train_loss_c, valid_loss_c, test_loss_c])
            
            # Compute metrics
            train_metric = get_clf_metrics(args, encoder, clf, train_loader, device)
            valid_metric = get_clf_metrics(args, encoder, clf, valid_loader, device)
            test_metric = get_clf_metrics(args, encoder, clf, test_loader, device)
            
            train_mm = train_metric[monitoring_metric]
            valid_mm = valid_metric[monitoring_metric]
            test_mm = test_metric[monitoring_metric]
            
            print(f'Epoch {epoch}: Train Metric = {train_mm:.4f}, Validation Metric = {valid_mm:.4f}, Test Metric = {test_mm:.4f}')
            metric_list.append([train_metric, valid_metric, test_metric])
            final_test_mm = test_mm
            epochs_trained = epoch
            
            # Learning rate scheduling
            encoder_scheduler.step(valid_mm)
            clf_scheduler.step(valid_mm)
            
            # if valid_loss_c < best_valid_loss:
            #     best_valid_loss = valid_loss_c
            if valid_mm > best_valid_mm:
                best_valid_mm = valid_mm
                early_stop_counter = 0
            else:
                early_stop_counter += 1
            
            # Check for early stopping
            if early_stop_counter >= patience:
                print(f'Early stopping triggered at epoch {epoch}')
                break
        
        # Save final results
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'wb') as f:
            pickle.dump([args, loss_list, metric_list], f)

        if final_test_mm is not None:
            run_name = os.path.basename(output_file)
            write_final_metric_row(summary_file, run_name, final_test_mm, epochs_trained)
            print(f'Final summary written: {run_name}, score={final_test_mm:.4f}, epochs={epochs_trained}')
        

    ## Run -- finetune -- baseline
    if torch.cuda.device_count() > 1:
        encoder = Encoder(args)
        encoder = nn.DataParallel(encoder).to(device)
        clf = Classifier(args)
        clf = nn.DataParallel(clf).to(device)
    else:
        encoder = Encoder(args).to(device)
        clf = Classifier(args).to(device)
    
    encoder_optimizer = torch.optim.Adam(encoder.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    clf_optimizer = torch.optim.Adam(clf.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    encoder_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(encoder_optimizer, mode='max', factor=0.5, patience=10)
    clf_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(clf_optimizer, mode='max', factor=0.5, patience=10)
    
    loss_list = []
    metric_list = []
    best_valid_mm = 0
    best_valid_loss = float('inf')
    # best_model_path = f'model_finetune/{args.data_name}/{args.data_name}_{args.seed}_{args.feature}_{args.loss_type}_{args.lam}_{k}_baseline.pth'
    output_file = f'out_finetune/{args.data_name}/{args.data_name}_pt-{pretrain_tag}_{args.feature}_{args.loss_type}_{args.lam}_{k}_baseline'
    
    # Early stopping parameters
    patience = 20
    early_stop_counter = 0
    
    # Check if output already exists
    if os.path.exists(output_file):
        print(f"Output file {output_file} already exists. Skipping this run.")
    else:
        final_test_mm = None
        epochs_trained = 0
        # Run experiment
        for epoch in range(1, args.epochs_finetune + 1):
            train_loss, train_loss_c = train(args, encoder, clf, encoder_optimizer, clf_optimizer, train_loader, mode='baseline', device=device)
            valid_loss, valid_loss_c = test(args, encoder, clf, valid_loader, mode='valid', device=device)
            test_loss, test_loss_c = test(args, encoder, clf, test_loader, mode='test', device=device)
            
            print(f'Epoch {epoch}: Train Loss = {train_loss:.4f}, Validation Loss = {valid_loss:.4f}, Test Loss = {test_loss:.4f}')
            print(f'Epoch {epoch}: Train Loss_c = {train_loss_c:.4f}, Validation Loss_c = {valid_loss_c:.4f}, Test Loss_c = {test_loss_c:.4f}')
            loss_list.append([train_loss, valid_loss, test_loss, train_loss_c, valid_loss_c, test_loss_c])
            
            # Compute metrics
            train_metric = get_clf_metrics(args, encoder, clf, train_loader, device)
            valid_metric = get_clf_metrics(args, encoder, clf, valid_loader, device)
            test_metric = get_clf_metrics(args, encoder, clf, test_loader, device)
            
            train_mm = train_metric[monitoring_metric]
            valid_mm = valid_metric[monitoring_metric]
            test_mm = test_metric[monitoring_metric]
            
            print(f'Epoch {epoch}: Train Metric = {train_mm:.4f}, Validation Metric = {valid_mm:.4f}, Test Metric = {test_mm:.4f}')
            metric_list.append([train_metric, valid_metric, test_metric])
            final_test_mm = test_mm
            epochs_trained = epoch
            
            # Learning rate scheduling
            encoder_scheduler.step(valid_mm)
            clf_scheduler.step(valid_mm)
            
            # if valid_loss_c < best_valid_loss:
            #     best_valid_loss = valid_loss_c
            if valid_mm > best_valid_mm:
                best_valid_mm = valid_mm
                early_stop_counter = 0
            else:
                early_stop_counter += 1
            
            # Check for early stopping
            if early_stop_counter >= patience:
                print(f'Early stopping triggered at epoch {epoch}')
                break
        
        # Save final results
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'wb') as f:
            pickle.dump([args, loss_list, metric_list], f)

        if final_test_mm is not None:
            run_name = os.path.basename(output_file)
            write_final_metric_row(summary_file, run_name, final_test_mm, epochs_trained)
            print(f'Final summary written: {run_name}, score={final_test_mm:.4f}, epochs={epochs_trained}')

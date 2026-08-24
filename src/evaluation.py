import torch
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.preprocessing import label_binarize


def repeat_if_batch_size_one(tensor):
    return torch.cat([tensor, tensor], dim=0) if tensor.size(0) == 1 else tensor


def get_clf_acc(args, encoder, clf, loader, device):
    """Quick accuracy-only evaluation (used by legacy probe script)."""
    metrics = get_clf_metrics(args, encoder, clf, loader, device)
    return metrics['accuracy']


def get_clf_metrics(args, encoder, clf, loader, device, subject_ids=None):
    """Compute classification metrics for EncoderNView + ClassifierNView.

    Batch format from Load_Dataset:
        (*views_orig[N], *views_aug[N], y)  — 2*N + 1 tensors.

    subject_ids: optional array-like, one id per sample, aligned index-for-index
        with `loader`'s (non-shuffled) iteration order. When given, also computes
        'f1_subject_macro' — the per-subject macro-F1 (sklearn default label set,
        i.e. classes present for that subject), averaged unweighted across subjects.
    """
    encoder.eval()
    clf.eval()
    num_views = encoder.num_views
    with torch.no_grad():
        logit_all, pred_all, y_all = [], [], []
        for batch in loader:
            batch      = [t.float().to(device) for t in batch]
            views_orig = [repeat_if_batch_size_one(v) for v in batch[:num_views]]
            y          = repeat_if_batch_size_one(batch[2 * num_views].long())

            hiddens, projs = encoder(*views_orig)
            inputs = projs if args.feature == 'latent' else hiddens
            logit  = clf(inputs)

            pred = logit.detach().argmax(dim=1)
            logit_all.append(logit)
            pred_all.append(pred)
            y_all.append(y)

        logit_all = torch.cat(logit_all, dim=0)
        pred_all  = torch.cat(pred_all,  dim=0)
        y_all     = torch.cat(y_all,     dim=0)

        y_true  = y_all.cpu().numpy()
        y_pred  = pred_all.cpu().numpy()
        y_score = torch.softmax(logit_all, dim=1).cpu().numpy()

        accuracy = (y_true == y_pred).mean()
        precision, recall, f1_macro, _ = precision_recall_fscore_support(
            y_true, y_pred, average='macro', zero_division=0)
        _, _, f1_weighted, _ = precision_recall_fscore_support(
            y_true, y_pred, average='weighted', zero_division=0)
        cm = confusion_matrix(y_true, y_pred)

        auroc = auprc = None
        if len(np.unique(y_true)) > 1:
            if args.num_target == 2:
                auroc = roc_auc_score(y_true, y_score[:, 1])
                auprc = average_precision_score(y_true, y_score[:, 1])
            else:
                y_true_bin = label_binarize(y_true, classes=range(args.num_target))
                try:
                    auroc = roc_auc_score(y_true_bin, y_score, average='macro', multi_class='ovr')
                    auprc = average_precision_score(y_true_bin, y_score, average='macro')
                except Exception:
                    pass
        else:
            print(f"Warning: Only one class present in the evaluation set "
                  f"(Class {np.unique(y_true)[0]}). AUROC and AUPRC are undefined.")

        f1_subject_macro, n_subjects = None, None
        if subject_ids is not None:
            subject_ids = np.asarray(subject_ids)
            if len(subject_ids) == len(y_true):
                per_subject_f1 = []
                for subj in np.unique(subject_ids):
                    mask = subject_ids == subj
                    _, _, f1_s, _ = precision_recall_fscore_support(
                        y_true[mask], y_pred[mask], average='macro', zero_division=0)
                    per_subject_f1.append(f1_s)
                f1_subject_macro = float(np.mean(per_subject_f1))
                n_subjects = len(per_subject_f1)
            else:
                print(f'Warning: subject_ids length ({len(subject_ids)}) != '
                      f'evaluation set size ({len(y_true)}); skipping f1_subject_macro.')

        return {
            'accuracy':         accuracy,
            'precision':        precision,
            'recall':           recall,
            'f1_macro':         f1_macro,
            'f1_weighted':      f1_weighted,
            'confusion_matrix': cm,
            'auroc':            auroc,
            'auprc':            auprc,
            'f1_subject_macro': f1_subject_macro,
            'n_subjects':       n_subjects,
        }


# Alias kept for any code that still imports the old nview name.
get_clf_metrics_nview = get_clf_metrics
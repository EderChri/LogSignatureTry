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


def get_clf_metrics(args, encoder, clf, loader, device):
    """Compute classification metrics for EncoderNView + ClassifierNView.

    Batch format from Load_Dataset:
        (*views_orig[N], *views_aug[N], y)  — 2*N + 1 tensors.
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
        precision, recall, f1, _ = precision_recall_fscore_support(
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

        return {
            'accuracy':         accuracy,
            'precision':        precision,
            'recall':           recall,
            'f1_score':         f1,
            'confusion_matrix': cm,
            'auroc':            auroc,
            'auprc':            auprc,
        }


# Alias kept for any code that still imports the old nview name.
get_clf_metrics_nview = get_clf_metrics
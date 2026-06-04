"""
describe_datasets.py — print a summary table for all preprocessed datasets.

Usage:
  python describe_datasets.py
  python describe_datasets.py --dir preprocessed_data --format tsv
"""

import argparse
import os
import pickle
import sys

import numpy as np


def describe_split(X, y, label):
    if X is None or (hasattr(X, '__len__') and len(X) == 0):
        return None
    n = len(X)
    if n == 0:
        return None
    classes = np.unique(y) if y is not None and len(y) > 0 else np.array([])
    n_cls = len(classes)
    if n_cls > 0:
        counts = np.bincount(y.astype(int))
        counts = counts[counts > 0]
        balance = f"min={counts.min()} max={counts.max()}"
    else:
        balance = "—"
    return {"split": label, "n": n, "n_cls": n_cls, "balance": balance}


def describe_file(path):
    fname = os.path.basename(path)
    tag = fname.replace(".pkl", "")
    size_mb = os.path.getsize(path) / 1024**2

    with open(path, "rb") as f:
        data = pickle.load(f)

    X_tr, _, _, y_tr, X_va, _, _, y_va, X_te, _, _, y_te = data

    # Determine shape from first non-empty split
    shape = None
    for X in (X_tr, X_va, X_te):
        if X is not None and hasattr(X, "shape") and len(X) > 0:
            shape = X.shape  # (N, D, L)
            dtype = X.dtype
            break

    if shape is None:
        return {"tag": tag, "error": "no data found"}

    _, D, L = shape

    splits = []
    for X, y, lbl in [(X_tr, y_tr, "train"), (X_va, y_va, "val"), (X_te, y_te, "test")]:
        info = describe_split(X, y, lbl)
        if info:
            splits.append(info)

    total_n = sum(s["n"] for s in splits)
    n_cls_list = [s["n_cls"] for s in splits if s["n_cls"] > 0]
    n_cls = max(n_cls_list) if n_cls_list else 0

    split_summary = "  |  ".join(
        f"{s['split']}: {s['n']:,} ({s['n']/max(total_n,1)*100:.0f}%)"
        for s in splits
    )
    balance_tr = next((s["balance"] for s in splits if s["split"] == "train"), "—")

    return {
        "tag": tag,
        "D": D,
        "L": L,
        "dtype": str(dtype),
        "n_cls": n_cls,
        "total_n": total_n,
        "splits": split_summary,
        "train_balance": balance_tr,
        "size_mb": size_mb,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default="preprocessed_data")
    parser.add_argument("--format", choices=["table", "tsv"], default="table")
    args = parser.parse_args()

    files = sorted(
        f for f in os.listdir(args.dir) if f.endswith(".pkl")
    )
    if not files:
        print(f"No .pkl files found in {args.dir}", file=sys.stderr)
        sys.exit(1)

    rows = []
    for fname in files:
        path = os.path.join(args.dir, fname)
        try:
            row = describe_file(path)
        except Exception as e:
            row = {"tag": fname.replace(".pkl", ""), "error": str(e)}
        rows.append(row)

    if args.format == "tsv":
        cols = ["tag", "D", "L", "dtype", "n_cls", "total_n", "splits", "train_balance", "size_mb"]
        print("\t".join(cols))
        for r in rows:
            if "error" in r:
                print(f"{r['tag']}\tERROR: {r['error']}")
                continue
            print("\t".join(str(r.get(c, "")) for c in cols))
        return

    # Pretty table
    col_w = {
        "tag":           36,
        "D/L":            8,
        "dtype":          8,
        "cls":            5,
        "total":          8,
        "splits":        52,
        "train_balance": 24,
        "MB":             7,
    }
    header = (
        f"{'Dataset':<{col_w['tag']}}"
        f"{'D×L':>{col_w['D/L']}}"
        f"{'dtype':>{col_w['dtype']}}"
        f"{'cls':>{col_w['cls']}}"
        f"{'total':>{col_w['total']}}"
        f"  {'splits':<{col_w['splits']}}"
        f"  {'train balance':<{col_w['train_balance']}}"
        f"{'MB':>{col_w['MB']}}"
    )
    sep = "-" * len(header)
    print(sep)
    print(header)
    print(sep)
    for r in rows:
        if "error" in r:
            print(f"{r['tag']:<{col_w['tag']}}  ERROR: {r['error']}")
            continue
        dl = f"{r['D']}×{r['L']}"
        print(
            f"{r['tag']:<{col_w['tag']}}"
            f"{dl:>{col_w['D/L']}}"
            f"{r['dtype']:>{col_w['dtype']}}"
            f"{r['n_cls']:>{col_w['cls']}}"
            f"{r['total_n']:>{col_w['total']},}"
            f"  {r['splits']:<{col_w['splits']}}"
            f"  {r['train_balance']:<{col_w['train_balance']}}"
            f"{r['size_mb']:>{col_w['MB']}.0f}"
        )
    print(sep)


if __name__ == "__main__":
    main()

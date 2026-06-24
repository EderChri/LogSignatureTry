"""
aggregate_results.py — Compute mean ± std across seeds for all summary TSV files.

For each final_test_metric_summary.tsv or final_pretrain_summary.tsv found under
out_finetune/ and out_pretrain/, strips the seed field from each run name and
produces an aggregated TSV next to the original.

Run name convention expected:
  <prefix>_ep<N>_<seed>[_<enc_suffix>][_<mode>]
                         ^^^^^
                         integer seed — this field is stripped for grouping.

Usage:
  python aggregate_results.py [--glob out_finetune out_pretrain]
"""

import re
import sys
import glob
import argparse
import statistics
from collections import defaultdict
from pathlib import Path


# ---------------------------------------------------------------------------
# Seed-stripping
# ---------------------------------------------------------------------------

_SEED_RE = re.compile(r'_ep(\d+)_(\d+)')


def strip_seed(name: str) -> tuple[str, str | None]:
    """Return (group_key, seed_str) for a run name.

    Matches the first occurrence of _ep<N>_<seed>, removes the _<seed> part.
    Returns (name, None) unchanged if the pattern isn't found.
    """
    m = _SEED_RE.search(name)
    if m is None:
        return name, None
    seed = m.group(2)
    group = name[:m.start()] + f'_ep{m.group(1)}' + name[m.end():]
    return group, seed


# ---------------------------------------------------------------------------
# TSV IO
# ---------------------------------------------------------------------------

def read_tsv(path: str) -> list[tuple[str, float, str]]:
    """Read summary TSV, skip header and comment lines.

    Returns list of (run_name, metric_value, extra_cols_str).
    """
    rows = []
    with open(path) as f:
        for line in f:
            line = line.rstrip('\n')
            if not line or line.startswith('#') or line.startswith('run_name'):
                continue
            parts = line.split('\t')
            if len(parts) < 2:
                continue
            try:
                metric = float(parts[1])
            except ValueError:
                continue
            extra = '\t'.join(parts[2:]) if len(parts) > 2 else ''
            rows.append((parts[0], metric, extra))
    return rows


def write_aggregated(path: str, rows: list[tuple], header: str):
    with open(path, 'w') as f:
        f.write(header + '\n')
        for row in rows:
            f.write('\t'.join(str(x) for x in row) + '\n')


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def aggregate_file(summary_path: str) -> str | None:
    """Aggregate one summary TSV.  Returns the output path, or None if skipped."""
    rows = read_tsv(summary_path)
    if not rows:
        return None

    # Detect whether this is a pretrain or finetune summary from the column name
    is_pretrain = 'pretrain' in summary_path.lower()
    metric_col = 'best_valid_loss' if is_pretrain else 'final_test_score'

    groups: dict[str, list[float]] = defaultdict(list)
    for name, value, _ in rows:
        group, seed = strip_seed(name)
        groups[group].append(value)

    agg_rows = []
    for group, values in sorted(groups.items()):
        n = len(values)
        mean = statistics.mean(values)
        std = statistics.pstdev(values) if n > 1 else 0.0
        agg_rows.append((group, f'{mean:.6f}', f'{std:.6f}', n))

    out_path = summary_path.replace('.tsv', '_agg.tsv')
    header = f'run_group\tmean_{metric_col}\tstd_{metric_col}\tnum_seeds'
    write_aggregated(out_path, agg_rows, header)
    return out_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Aggregate multi-seed summary TSVs')
    parser.add_argument('--glob', nargs='+', default=['out_finetune', 'out_pretrain'],
                        help='Root directories to search for summary TSVs')
    args = parser.parse_args()

    patterns = [
        f'{root}/**/final_test_metric_summary.tsv'
        for root in args.glob
    ] + [
        f'{root}/**/final_pretrain_summary.tsv'
        for root in args.glob
    ]

    found = []
    for pattern in patterns:
        found.extend(glob.glob(pattern, recursive=True))

    if not found:
        print('No summary TSV files found.  Run experiments first.')
        sys.exit(0)

    for path in sorted(found):
        out = aggregate_file(path)
        if out:
            print(f'Aggregated: {path}  →  {out}')
        else:
            print(f'Skipped (empty): {path}')


if __name__ == '__main__':
    main()

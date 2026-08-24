"""
plot_wilcoxon_holm_era_a.py — Second view on the Era A sweep (see
plot_critical_difference_era_a.py for scope/data details): instead of average
ranks + a single Nemenyi critical distance, this runs all pairwise Wilcoxon
signed-rank tests (paired by pretrain seed) and Holm-corrects across every
comparison, then draws the clique bars from the actual pairwise significance
graph (maximal cliques of "not significantly different" edges) rather than a
rank-distance threshold.

Usage: python vis/plot_wilcoxon_holm_era_a.py
"""

import sys
from itertools import combinations

import networkx as nx
import numpy as np
from scipy import stats
from statsmodels.stats.multitest import multipletests

sys.path.insert(0, 'vis')
from plot_critical_difference import plot_cd_diagram_many
from plot_critical_difference_era_a import build_matrix

ALPHA = 0.05


def pairwise_wilcoxon_holm(data, alpha=ALPHA):
    """Return (reject, p_holm) arrays aligned with combinations(range(k), 2)."""
    n, k = data.shape
    pairs = list(combinations(range(k), 2))
    pvals = np.empty(len(pairs))
    for idx, (i, j) in enumerate(pairs):
        diff = data[:, i] - data[:, j]
        if np.all(diff == 0):
            pvals[idx] = 1.0
        else:
            _, pvals[idx] = stats.wilcoxon(data[:, i], data[:, j])
    reject, p_holm, _, _ = multipletests(pvals, alpha=alpha, method='holm')
    return pairs, reject, pvals, p_holm


def cliques_from_significance(methods, sorted_order, pairs, reject):
    """Build the 'not significantly different' graph over methods (using
    original indices), find maximal cliques, then map each clique to the
    (min, max) index range in rank-sorted order for bar drawing."""
    k = len(methods)
    pos_in_sorted = {orig_i: s for s, orig_i in enumerate(sorted_order)}

    g = nx.Graph()
    g.add_nodes_from(range(k))
    for (i, j), rej in zip(pairs, reject):
        if not rej:   # not significantly different -> connected
            g.add_edge(i, j)

    groups = []
    for clique in nx.find_cliques(g):
        if len(clique) < 2:
            continue
        sorted_positions = [pos_in_sorted[c] for c in clique]
        groups.append((min(sorted_positions), max(sorted_positions)))
    # drop groups fully contained in another, de-duplicate
    maximal = []
    for gI in set(groups):
        if not any(gI != h and h[0] <= gI[0] and gI[1] <= h[1] for h in groups):
            maximal.append(gI)
    return sorted(maximal)


def main():
    methods, data = build_matrix()
    n, k = data.shape

    ranks = np.zeros_like(data)
    for i in range(n):
        ranks[i] = stats.rankdata(-data[i])
    avg_ranks = dict(zip(methods, ranks.mean(axis=0)))

    order = np.argsort([avg_ranks[m] for m in methods])

    pairs, reject, pvals, p_holm = pairwise_wilcoxon_holm(data)
    n_pairs = len(pairs)
    n_raw_sig = int((pvals < ALPHA).sum())
    n_holm_sig = int(reject.sum())

    print(f'k={k} methods, n={n} paired seeds, {n_pairs} pairwise comparisons')
    print(f'  raw p<{ALPHA}: {n_raw_sig}/{n_pairs}')
    print(f'  Holm-significant: {n_holm_sig}/{n_pairs}')
    print(f'  min raw p={pvals.min():.5f}, min Holm-adjusted p={p_holm.min():.4f}')

    groups = cliques_from_significance(methods, order, pairs, reject)
    print(f'  {len(groups)} maximal clique(s) of mutually non-significant methods')

    stats_line = (
        f'Wilcoxon signed-rank, Holm-corrected (α=0.05): {n_pairs} pairwise comparisons, '
        f'{n_raw_sig} raw p<0.05, {n_holm_sig} survive Holm correction  (k={k} methods, n={n} paired seeds)'
    )
    subtitle = (
        'Thick bars = maximal cliques of pairs not significantly different after Holm'
        ' correction (not a rank-distance threshold).'
    )
    if n_holm_sig == 0:
        subtitle += (
            f'\nNo pair survives correction at this n — min achievable exact Wilcoxon p at n={n} is '
            f'{1/2**(n-1):.4f}, below the Holm threshold needed for {n_pairs} comparisons.'
        )

    plot_cd_diagram_many(
        avg_ranks, None, k, n,
        title='Wilcoxon signed-rank + Holm — capture24 -> Opportunity, Era A sweep',
        out_path='plots/wilcoxon_holm_opportunity_era_a.png',
        stats_line=stats_line,
        subtitle=subtitle,
        groups=groups,
    )


if __name__ == '__main__':
    main()

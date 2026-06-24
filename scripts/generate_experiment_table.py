"""Generate experiment_table.md from both finetune TSVs (current + pre_canada_old).

Run: python generate_experiment_table.py
Output: experiment_table.md
"""

import pandas as pd
import re
import numpy as np
from collections import defaultdict

# ── Load all modes (finetune, freeze, baseline) ────────────────────────────
dfs = []
for path in [
    "out_finetune/_DA_HAR70plus_256_00/final_test_metric_summary.tsv",
    "out_finetune_pre_canada_old/_DA_HAR70plus_256_00/final_test_metric_summary.tsv",
]:
    try:
        df2 = pd.read_csv(path, sep='\t', comment='#')
        df2.columns = ['run_name', 'score', 'epochs']
        dfs.append(df2)
    except FileNotFoundError:
        print(f"Warning: {path} not found, skipping.")

df = pd.concat(dfs, ignore_index=True)

# Keep only HARTH→HAR70+ runs
df = df[df['run_name'].str.contains('pt-_DA_HARTH_256_00')].copy()

# ── Regex patterns ─────────────────────────────────────────────────────────
# 3-view: v2{v2}_v3{v3}_ep{ep}_{seed}[{enc}{lsig}{il}{cv}]_{feature}_{loss}_{lam}_{ft_seed}_{mode}
THREEVIEW_RE = re.compile(
    r'_v2(?P<v2>[\w]+?)_v3(?P<v3>[\w]+?)_ep(?P<ep>\d+)_(?P<pt_seed>\d+)'
    r'(?P<rest>.*?)'
    r'_(?P<feature>hidden|latent)_(?P<loss>ALL|TDF|TD|TF|DF|T|D|F)_(?P<lam>[\d.]+)_(?P<ft_seed>\d+)_(?P<mode>finetune|freeze|baseline)$'
)
# nview (2-view): v2{v2}_nview_ep{ep}_{seed}[{enc}{lsig}{il}]_{feature}_{loss}_{lam}_0_{mode}
NVIEW_RE = re.compile(
    r'_v2(?P<v2>[\w]+?)_nview_ep(?P<ep>\d+)_(?P<pt_seed>\d+)'
    r'(?P<rest>.*?)'
    r'_(?P<feature>hidden|latent)_(?P<loss>ALL|TDF|TD|TF|DF|T|D|F)_(?P<lam>[\d.]+)_0_(?P<mode>finetune|freeze|baseline)$'
)


def parse_rest(rest):
    """Parse enc_suffix + lsig_suffix + il_suffix + cv_suffix from middle segment."""
    enc = 'mlp_logsig' if rest.startswith('_mlp_logsig') else 'transformer'
    s = rest.removeprefix('_mlp_logsig')

    lsig_mode, lsig_win, lsig_smooth = 'stream', None, None
    stride, depth, lsn, gt = 1, 2, 0.0, False

    m = re.match(r'^_win(\d+)', s)
    if m:
        lsig_mode = 'window'; lsig_win = int(m.group(1)); s = s[m.end():]
    else:
        m = re.match(r'^_(tukey|ema)(\d+)', s)
        if m:
            lsig_mode = 'window_smooth'; lsig_smooth = m.group(1)
            lsig_win = int(m.group(2)); s = s[m.end():]
            m2 = re.match(r'^_sp([\d.]+)', s)
            if m2:
                s = s[m2.end():]
            m2 = re.match(r'^_msp(\d+)', s)
            if m2:
                s = s[m2.end():]

    m = re.match(r'^_s(\d+)', s)
    if m: stride = int(m.group(1)); s = s[m.end():]
    if s.startswith('_gt'): gt = True; s = s[3:]
    m = re.match(r'^_p[a-z]+', s)
    if m: s = s[m.end():]
    m = re.match(r'^_d(\d+)', s)
    if m: depth = int(m.group(1)); s = s[m.end():]
    m = re.match(r'^_lsn([\d.]+)', s)
    if m: lsn = float(m.group(1)); s = s[m.end():]

    il = 'attention'
    m = re.match(r'^_il([a-z]+)', s)   # [a-z]+ stops before _ so won't eat _cv
    if m: il = m.group(1); s = s[m.end():]

    cv = None
    m = re.match(r'^_cv([\d.]*)', s)
    if m:
        cv_val = m.group(1); cv = cv_val if cv_val else '?'; s = s[m.end():]

    return dict(enc=enc, lsig_mode=lsig_mode, lsig_win=lsig_win,
                lsig_smooth=lsig_smooth, stride=stride, depth=depth,
                lsn=lsn, gt=gt, il=il, cv=cv)


# ── Parse all rows ─────────────────────────────────────────────────────────
rows = []
for _, row in df.iterrows():
    rn = row['run_name']
    nview = False
    m = THREEVIEW_RE.search(rn)
    if not m:
        m = NVIEW_RE.search(rn)
        nview = True
    if not m:
        continue
    g = m.groupdict()
    p = parse_rest(g['rest'])
    rows.append({
        'nview': nview,
        'v2': g['v2'], 'v3': g.get('v3', '—'),
        'ep': int(g['ep']), 'pt_seed': int(g['pt_seed']),
        'feature': g['feature'], 'ft_seed': int(g.get('ft_seed', 0)),
        'mode': g['mode'],
        'score': row['score'],
        **p,
    })

parsed = pd.DataFrame(rows)
print(f"Parsed {len(parsed)} rows from {len(df)} total")

# ── Helpers ────────────────────────────────────────────────────────────────

def fmt_score(scores):
    """Return mean ± std string, or single value if n=1."""
    n = len(scores)
    if n == 0:
        return '—'
    mean = np.mean(scores)
    if n == 1:
        return f'{mean:.3f}'
    std = np.std(scores, ddof=1)
    return f'{mean:.3f}±{std:.3f}'


def get_scores(sub, mode):
    """Get per-seed scores for a given mode, deduplicating by pt_seed+ft_seed."""
    m = sub[sub['mode'] == mode]
    # drop duplicate (pt_seed, ft_seed) pairs, keep first
    m = m.drop_duplicates(subset=['pt_seed', 'ft_seed'])
    # only include ep=2 runs (exclude 200-epoch outliers)
    m = m[m['ep'] == 2]
    return m['score'].tolist()


def make_aug_str(cv, lsn):
    parts = []
    if cv is not None:
        parts.append(f'cv{cv}')
    if lsn > 0:
        parts.append(f'lsn{lsn}')
    return ', '.join(parts) if parts else '—'


def make_smooth_str(lsig_smooth, lsig_win, lsig_mode):
    if lsig_mode == 'window_smooth':
        return lsig_smooth or 'tukey'
    return '—'


def il_abbrev(il):
    return {'attention': 'att', 'bilinear': 'bil', 'viewembed': 'vemb',
            'crosstime': 'crt'}.get(il, il)


# ── Group into config rows ─────────────────────────────────────────────────
GROUP_KEYS = ['nview', 'v2', 'v3', 'enc', 'lsig_mode', 'lsig_win', 'lsig_smooth',
              'stride', 'depth', 'lsn', 'gt', 'il', 'cv']

groups = parsed.groupby(GROUP_KEYS, dropna=False)

table_rows = []
for key, sub in groups:
    d = dict(zip(GROUP_KEYS, key))
    ft = get_scores(sub, 'finetune')
    fr = get_scores(sub, 'freeze')
    bl = get_scores(sub, 'baseline')
    n = max(len(ft), len(fr), len(bl))
    if n == 0:
        continue
    d['n'] = n
    d['ft'] = fmt_score(ft)
    d['fr'] = fmt_score(fr)
    d['bl'] = fmt_score(bl)
    table_rows.append(d)

tbl = pd.DataFrame(table_rows)
print(f"Config groups: {len(tbl)}")


# ── Generate Markdown ──────────────────────────────────────────────────────
lines = []
lines.append("# HARTH → HAR70+ Experiment Table\n")
lines.append("Pretrain: HARTH (2 epochs unless noted). Finetune: HAR70+.")
lines.append("Scores = accuracy (mean ± std across seeds, finetune mode unless noted).")
lines.append("")
lines.append("**Legend:** Enc = encoder (T=transformer, M=mlp_logsig);")
lines.append("IL = interaction layer (att=attention/default, bil=bilinear, vemb=view_embed, crt=cross_time);")
lines.append("LS = log-signature mode (str=stream, W=window, WS=window_smooth);")
lines.append("Win = window size; D = logsig depth; Sm = smoothing (tky=tukey, ema);")
lines.append("Aug = augmentation (cv=cross-view logsig λ, lsn=per-level noise scale);")
lines.append("n = number of seeds; FT = finetune; FR = freeze; BL = random-init baseline.")
lines.append("")

def section(title, mask, sort_keys):
    sub = tbl[mask].sort_values(sort_keys)
    if sub.empty:
        return
    lines.append(f"## {title}\n")
    lines.append("| Enc | IL | LS | Win | D | Sm | Stride | Aug | n | FT | FR | BL | Notes |")
    lines.append("|-----|----|----|-----|---|----|--------|-----|---|----|----|----|----|")
    for _, r in sub.iterrows():
        enc = 'M' if r['enc'] == 'mlp_logsig' else 'T'
        il  = il_abbrev(str(r['il']) if pd.notna(r['il']) else 'attention')
        lm  = {'stream': 'str', 'window': 'W', 'window_smooth': 'WS'}.get(str(r['lsig_mode']), str(r['lsig_mode']))
        win = str(int(r['lsig_win'])) if pd.notna(r['lsig_win']) else '—'
        d   = str(int(r['depth'])) if pd.notna(r['depth']) else '2'
        sm  = make_smooth_str(r['lsig_smooth'], r['lsig_win'], r['lsig_mode'])
        st  = str(int(r['stride'])) if pd.notna(r['stride']) and int(r['stride']) != 1 else '—'
        cv_val = str(r['cv']) if pd.notna(r['cv']) else None
        lsn_val = float(r['lsn']) if pd.notna(r['lsn']) else 0.0
        aug = make_aug_str(cv_val, lsn_val)
        n   = str(int(r['n']))
        lines.append(f"| {enc} | {il} | {lm} | {win} | {d} | {sm} | {st} | {aug} | {n} | {r['ft']} | {r['fr']} | {r['bl']} |  |")
    lines.append("")


# Section A: xt + dx + xf  (3-view transformer)
section(
    "A. xt + dx + xf (3-view, Transformer)",
    (tbl['nview'] == False) & (tbl['v2'] == 'dx') & (tbl['v3'] == 'xf'),
    ['il', 'enc']
)

# Section B: xt + dx + logsig (3-view)
section(
    "B. xt + dx + logsig (3-view)",
    (tbl['nview'] == False) & (tbl['v2'] == 'dx') & (tbl['v3'] == 'logsig'),
    ['enc', 'il', 'lsig_mode', 'lsig_win', 'cv']
)

# Section C: xt + logsig nview (2-view)
section(
    "C. xt + logsig (2-view nview)",
    (tbl['nview'] == True) & (tbl['v2'] == 'logsig'),
    ['enc', 'il', 'lsig_mode', 'lsig_win']
)

# Section D: xt + logsig + xf (3-view, logsig as v2)
section(
    "D. xt + logsig + xf (3-view, logsig as v2)",
    (tbl['nview'] == False) & (tbl['v2'] == 'logsig') & (tbl['v3'] == 'xf'),
    ['enc', 'il', 'lsig_mode', 'lsig_win']
)

# Section E: xt + dx only (no logsig) — 2-view nview
section(
    "E. xt + dx (2-view nview)",
    (tbl['nview'] == True) & (tbl['v2'] == 'dx'),
    ['il', 'enc']
)

# Catch remaining
shown_mask = (
    ((tbl['nview'] == False) & (tbl['v2'] == 'dx') & (tbl['v3'] == 'xf')) |
    ((tbl['nview'] == False) & (tbl['v2'] == 'dx') & (tbl['v3'] == 'logsig')) |
    ((tbl['nview'] == True) & (tbl['v2'] == 'logsig')) |
    ((tbl['nview'] == False) & (tbl['v2'] == 'logsig') & (tbl['v3'] == 'xf')) |
    ((tbl['nview'] == True) & (tbl['v2'] == 'dx'))
)
other = tbl[~shown_mask]
if not other.empty:
    section("F. Other", ~shown_mask, ['nview', 'v2', 'v3', 'enc'])

md = '\n'.join(lines)
with open('experiment_table.md', 'w') as f:
    f.write(md)

print("Written experiment_table.md")
print(f"Sections: {len([l for l in lines if l.startswith('##')])}")
